"""
src/orchestrator.py

Orchestrator agent for ContractLens MS3 — Qwen 3.5-27B on OpenRouter, three
function-calling tools (retrieve, lookup_hypothesis, run_full_analysis), and
SSE streaming surfaced as an AsyncIterator the TUI consumes token-by-token.

Streamed event shape (`OrchestratorEvent.kind`):
    "think"          — reasoning content from the model (delta.reasoning)
    "content"        — assistant content tokens (delta.content)
    "tool_call"      — model has asked to call a tool (after stream completes)
    "tool_result"    — local handler ran; output appended to the conversation
    "turn_complete"  — model returned content with no further tool calls

The function-calling loop reinjects tool outputs as `role=tool` messages and
re-streams until the model produces content-only — capped at MAX_TOOL_ITERS
turns to stop runaway loops.

Dependencies:
  - src/tools.py for TOOL_SCHEMAS + TOOL_HANDLERS (must be set_tool_context'd
    by the caller before `run_turn` is awaited).
  - OpenRouter API key in env (OPENROUTER_API_KEY) or passed at construction.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from src.loaders._constants import (
    OPENROUTER_BASE_URL,
    OPENROUTER_ORCHESTRATOR_ID,
)
from src.tools import TOOL_HANDLERS, TOOL_SCHEMAS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


MAX_TOOL_ITERS = 8       # guardrail against the model looping on tool calls
DEFAULT_TIMEOUT = 300.0  # OpenRouter SSE timeout, seconds


SYSTEM_PROMPT = """\
You are ContractLens, an expert NDA-review assistant.

Tools available to you:

1. retrieve(query, mode?, top_k?, hypothesis_id?, label_filter?)
   - Pulls example spans from a training corpus of other NDAs.
   - These are REASONING AIDS ONLY. You may use them to understand clause
     patterns or compare phrasings, but you MUST NOT cite them as evidence
     and MUST NOT quote them as text from the analyzed contract.

2. lookup_hypothesis(h_id)
   - Reads a cached per-hypothesis result (label, evidence_spans,
     verbatim_quote, confidence) from a prior run_full_analysis in this
     session. Returns null if not yet computed.

3. run_full_analysis(contract_id, retrieval_mode?)
   - Runs the heavy 17-hypothesis analysis. REQUIRES USER APPROVAL.
     The user will be prompted to confirm before this fires.
   - Only call this when the user explicitly requests a full review,
     types /analyze, or the question genuinely cannot be answered
     without all 17 labels.

Decision rules — pick the lightest path that answers the user:
  - Direct factual question about the contract → answer from the
    [CONTRACT] block directly; cite specific clauses verbatim.
  - "What does this remind you of?" / pattern reasoning → call retrieve.
  - "Is hypothesis H06 entailed?" / asking about a known label →
    call lookup_hypothesis. If it returns null, decide whether to ask
    the user about run_full_analysis or answer from context.
  - "Run the full review" / "Analyze this NDA" → call run_full_analysis.

Citations:
  - Final evidence MUST come from the analyzed contract — quote verbatim.
  - Distinguish [contract] and [external] sources when both are referenced.
  - Never attribute a quote to the contract that doesn't appear in it.
"""


# ── Event shape the TUI consumes ────────────────────────────────────────────


@dataclass
class OrchestratorEvent:
    """One streamed event. The TUI / harness matches on .kind."""
    kind: str
    text: Optional[str] = None              # for "think" / "content"
    tool_name: Optional[str] = None         # for "tool_call" / "tool_result"
    tool_arguments: Optional[dict] = None   # for "tool_call"
    tool_output: Optional[object] = None    # for "tool_result"


# ── Orchestrator ────────────────────────────────────────────────────────────


@dataclass
class Orchestrator:
    """
    Stateless per-turn driver. Conversation history is passed in by the
    caller (M4 owns the session); the orchestrator only knows about a
    single turn at a time.
    """
    model_id: str = OPENROUTER_ORCHESTRATOR_ID
    base_url: str = OPENROUTER_BASE_URL
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = DEFAULT_TIMEOUT
    enable_thinking: bool = True
    http_referer: Optional[str] = None
    x_title: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to .env or export it "
                "before constructing Orchestrator."
            )

    # ── Public API ────────────────────────────────────────────────────────

    async def run_turn(
        self,
        user_message: str,
        history: List[dict],
        contract_block: Optional[str] = None,
    ) -> AsyncIterator[OrchestratorEvent]:
        """
        Run one full user turn — possibly multiple model calls if the model
        chooses to use tools — and stream events as they arrive.

        `history` is a list of {role, content} dicts (and optionally assistant
        messages with tool_calls). The user's new message is appended inside;
        the caller should append the resulting assistant message to history
        themselves once the iterator completes.
        """
        wrapped_user = _wrap_user_message(user_message, contract_block)
        messages: List[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": wrapped_user},
        ]

        for _ in range(MAX_TOOL_ITERS):
            think_buf: list[str] = []
            content_buf: list[str] = []
            tool_calls_acc: dict[int, dict] = {}

            async for delta in self._stream(messages):
                kind = delta["type"]
                if kind == "think":
                    think_buf.append(delta["text"])
                    yield OrchestratorEvent(kind="think", text=delta["text"])
                elif kind == "content":
                    content_buf.append(delta["text"])
                    yield OrchestratorEvent(kind="content", text=delta["text"])
                elif kind == "tool_call_delta":
                    _merge_tool_call_delta(tool_calls_acc, delta["delta"])

            tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
            assistant_msg: dict = {"role": "assistant", "content": "".join(content_buf)}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                yield OrchestratorEvent(kind="turn_complete")
                return

            # Dispatch each tool call and append its result message.
            for tc in tool_calls:
                name = tc["function"]["name"]
                raw_args = tc["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    args = {"_raw_arguments": raw_args}

                yield OrchestratorEvent(
                    kind="tool_call", tool_name=name, tool_arguments=args,
                )

                output = await _dispatch_tool(name, args)

                yield OrchestratorEvent(
                    kind="tool_result", tool_name=name, tool_output=output,
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id") or "",
                    "name": name,
                    "content": _safe_json_dumps(output),
                })

        # Hit the iteration cap — surface a turn_complete so the TUI doesn't hang.
        yield OrchestratorEvent(kind="turn_complete")

    # ── SSE streaming (urllib on a worker thread → asyncio.Queue) ─────────

    async def _stream(self, messages: List[dict]) -> AsyncIterator[dict]:
        """
        Stream chunks from OpenRouter chat/completions and yield decoded
        delta events: {"type": "think"|"content"|"tool_call_delta", ...}.

        Implemented by running urllib on a worker thread (urllib is sync)
        and piping decoded chunks back through an asyncio.Queue so the
        caller can interleave with other async work.
        """
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": True,
            "temperature": self.temperature,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
        }
        if not self.enable_thinking:
            payload["reasoning"] = {"exclude": True}

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def _push(item) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def reader() -> None:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}

                        rc = delta.get("reasoning") or delta.get("reasoning_content")
                        if rc:
                            _push({"type": "think", "text": rc})

                        content = delta.get("content")
                        if content:
                            _push({"type": "content", "text": content})

                        for tc_delta in (delta.get("tool_calls") or []):
                            _push({"type": "tool_call_delta", "delta": tc_delta})
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace") if e.fp else ""
                _push({"type": "content",
                       "text": f"\n[orchestrator HTTPError {e.code}: {body[:500]}]\n"})
            except Exception as e:
                _push({"type": "content",
                       "text": f"\n[orchestrator stream error: {e!r}]\n"})
            finally:
                _push(SENTINEL)

        loop.run_in_executor(None, reader)

        while True:
            item = await queue.get()
            if item is SENTINEL:
                return
            yield item


# ── Helpers ─────────────────────────────────────────────────────────────────


def _wrap_user_message(user_message: str, contract_block: Optional[str]) -> str:
    if not contract_block:
        return user_message
    return (
        "[CONTRACT]\n"
        f"{contract_block}\n"
        "[/CONTRACT]\n\n"
        "[QUESTION]\n"
        f"{user_message}\n"
        "[/QUESTION]"
    )


def _merge_tool_call_delta(acc: dict, tc_delta: dict) -> None:
    """
    OpenAI/OpenRouter stream tool_call fragments piece-by-piece, keyed by
    `index`. Accumulate id + function.name + function.arguments across deltas.
    """
    idx = tc_delta.get("index", 0)
    slot = acc.setdefault(idx, {
        "id": "",
        "type": "function",
        "function": {"name": "", "arguments": ""},
    })
    if tc_delta.get("id"):
        slot["id"] = tc_delta["id"]
    if tc_delta.get("type"):
        slot["type"] = tc_delta["type"]
    fn = tc_delta.get("function") or {}
    if fn.get("name"):
        slot["function"]["name"] += fn["name"]
    if fn.get("arguments"):
        slot["function"]["arguments"] += fn["arguments"]


async def _dispatch_tool(name: str, arguments: dict) -> object:
    """Call the registered handler; coerce errors into a structured result."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name!r}"}
    try:
        result = handler(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception as e:
        return {"error": repr(e)}


def _safe_json_dumps(obj: object) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps({"_repr": repr(obj)})
