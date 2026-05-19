"""
src/runtrace_recorder.py

Cross-cutting tool-call recorder. Every tool invocation by every agent passes
through this module so the audit chain required by MS3 req (h) is complete.

Two usage patterns:

1) Context manager — wrap a block that calls a tool:

       from src.runtrace_recorder import record_tool_call

       with record_tool_call("retrieve", {"query": q, "mode": "vector"},
                             agent_id="orchestrator") as call:
           result = retrieve(q, mode="vector")
           call.set_output(result)

2) Decorator — wrap a tool handler so every call is logged automatically:

       from src.runtrace_recorder import recorded_tool

       @recorded_tool("retrieve")
       def retrieve(query, mode, top_k=5, agent_id="orchestrator", ...):
           ...

The active recorder is bound via `set_active_recorder()`. The TUI / eval
harness sets it once at session start; Members 2/3/4 just call the helpers
above without threading a recorder argument through every layer.

A ContextVar backs the active recorder so concurrent asyncio tasks (M2's
worker pool of 5) each see the recorder bound to their parent context.
"""
from __future__ import annotations

import contextvars
import functools
import inspect
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Iterator, List, Optional, Tuple

from src.types import ToolCallRecord


# ── Active recorder (ContextVar so async fan-out sees the right one) ─────────

_active_recorder: contextvars.ContextVar[Optional["RunTraceRecorder"]] = (
    contextvars.ContextVar("contract_lens_active_recorder", default=None)
)


def set_active_recorder(recorder: Optional["RunTraceRecorder"]) -> None:
    """Bind (or unbind) the recorder visible to record_tool_call / @recorded_tool."""
    _active_recorder.set(recorder)


def get_active_recorder() -> Optional["RunTraceRecorder"]:
    return _active_recorder.get()


@contextmanager
def use_recorder(recorder: Optional["RunTraceRecorder"]) -> Iterator[None]:
    """Scoped binding — restores the previous active recorder on exit."""
    token = _active_recorder.set(recorder)
    try:
        yield
    finally:
        _active_recorder.reset(token)


# ── Recorder ─────────────────────────────────────────────────────────────────


class RunTraceRecorder:
    """
    Thread-safe in-memory accumulator for tool-call events during one session.

    The TUI's session loop owns one instance for the whole REPL. M2/M3's
    hypothesis pipeline can either share the session recorder or open a fresh
    one for a single contract analysis — both are supported.
    """

    def __init__(self) -> None:
        self._events: List[ToolCallRecord] = []
        self._counts: dict[Tuple[str, str], int] = {}
        self._lock = Lock()

    def push(self, record: ToolCallRecord) -> None:
        with self._lock:
            self._events.append(record)

    def next_count(self, agent_id: str, name: str) -> int:
        """Per-(agent_id, name) call ordinal, 1-indexed. Required by the schema."""
        with self._lock:
            key = (agent_id, name)
            self._counts[key] = self._counts.get(key, 0) + 1
            return self._counts[key]

    def snapshot(self) -> List[ToolCallRecord]:
        """Copy of all events accumulated so far. Safe to call mid-session."""
        with self._lock:
            return list(self._events)

    def events_for_agent(self, agent_id: str) -> List[ToolCallRecord]:
        with self._lock:
            return [e for e in self._events if e.get("agent_id") == agent_id]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._counts.clear()


# ── ToolCall context manager ─────────────────────────────────────────────────


@dataclass
class ToolCallContext:
    """Handle yielded by `record_tool_call`. Caller fills .set_output(...)."""
    agent_id: str
    name: str
    arguments: dict
    started_at: str
    _start_perf: float
    _output: Any = None
    _output_set: bool = False
    _error: Optional[str] = None

    def set_output(self, output: Any) -> None:
        self._output = output
        self._output_set = True

    def set_error(self, message: str) -> None:
        self._error = message


def _utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@contextmanager
def record_tool_call(
    name: str,
    arguments: dict,
    agent_id: str = "orchestrator",
) -> Iterator[ToolCallContext]:
    """
    Open-ended recorder block. Caller fills .set_output(...) before exit.

    Always safe: if no recorder is active (e.g. unit tests), the block runs
    normally and nothing is recorded.
    """
    ctx = ToolCallContext(
        agent_id=agent_id,
        name=name,
        arguments=arguments,
        started_at=_utc_iso(),
        _start_perf=time.perf_counter(),
    )
    try:
        yield ctx
    except BaseException as exc:
        ctx.set_error(repr(exc))
        _commit(ctx)
        raise
    _commit(ctx)


def _commit(ctx: ToolCallContext) -> None:
    duration_ms = (time.perf_counter() - ctx._start_perf) * 1000.0
    recorder = get_active_recorder()
    if recorder is None:
        return
    count = recorder.next_count(ctx.agent_id, ctx.name)
    record: ToolCallRecord = {
        "agent_id": ctx.agent_id,
        "name": ctx.name,
        "arguments": _safe_json(ctx.arguments),
        "count": count,
        "started_at": ctx.started_at,
        "duration_ms": round(duration_ms, 3),
    }
    if ctx._output_set:
        record["output"] = _safe_json(ctx._output)
    if ctx._error is not None:
        record["error"] = ctx._error
    recorder.push(record)


def _safe_json(obj: Any) -> Any:
    """
    Coerce to JSON-serialisable. Unknown types fall back to repr() so a
    misbehaving tool output can never break audit logging.
    """
    try:
        json.dumps(obj, default=str)
        return obj
    except (TypeError, ValueError):
        return repr(obj)


# ── Decorator form ───────────────────────────────────────────────────────────


def recorded_tool(name: str, agent_id_arg: str = "agent_id") -> Callable:
    """
    Decorator. Logs every call to the wrapped function.

    The decorated function should accept `agent_id` as a kwarg (defaults to
    "orchestrator" if omitted). Pass `agent_id_arg="<other_param>"` to source
    the agent id from a different parameter name.

    Works on both sync and async callables.
    """

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        is_coro = inspect.iscoroutinefunction(func)

        def _bind_args(args: tuple, kwargs: dict) -> Tuple[str, dict]:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            agent_id = bound.arguments.get(agent_id_arg, "orchestrator")
            arguments = {k: v for k, v in bound.arguments.items() if k != "self"}
            return agent_id, arguments

        if is_coro:
            @functools.wraps(func)
            async def awrapper(*args, **kwargs):
                agent_id, arguments = _bind_args(args, kwargs)
                with record_tool_call(name, arguments, agent_id=agent_id) as call:
                    output = await func(*args, **kwargs)
                    call.set_output(output)
                    return output
            return awrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            agent_id, arguments = _bind_args(args, kwargs)
            with record_tool_call(name, arguments, agent_id=agent_id) as call:
                output = func(*args, **kwargs)
                call.set_output(output)
                return output

        return wrapper

    return decorator
