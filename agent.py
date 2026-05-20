#!/usr/bin/env python3
"""
agent.py — ContractLens MS3 REPL entry point.

Usage:
    python agent.py --contract data/test.json --idx 0           # REPL, user mode
    python agent.py --contract data/test.json --idx 0 --dev     # REPL, developer mode
    python agent.py --contract data/test.json --idx 0 \\
                    --prompt "Who are the parties?"              # one-shot, no REPL
    python agent.py --eval                                       # headless batch (M5)

Environment:
    OPENROUTER_API_KEY — required for live inference (set in .env)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.runtrace import (
    build_session_runtrace,
    utc_now_iso,
    write_session_runtrace,
)
from src.runtrace_recorder import RunTraceRecorder, set_active_recorder
from src.session import SessionState
from src.tui import TUI
from src.types import ApprovalEvent


# ── Orchestrator import with stub fallback ────────────────────────────────────
#
# M1's src/orchestrator.py lands on integration day. Until then the stub below
# keeps the REPL fully functional for testing session/TUI/RunTrace machinery.

try:
    from src.orchestrator import (  # type: ignore[import]
        ApprovalRequestEvent,
        Orchestrator,
        StatusEvent,
        ThinkEvent,
        TokenEvent,
        ToolCallEvent,
        TurnCompleteEvent,
    )
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False


# ── Stub orchestrator (active until M1 lands) ─────────────────────────────────

class _StubTokenEvent:
    def __init__(self, text: str) -> None:
        self.text = text
        self.is_thinking = False


class _StubTurnCompleteEvent:
    def __init__(self) -> None:
        self.citations: dict = {"contract": [], "external": []}


class _StubOrchestrator:
    """
    Echo-back stub so the REPL can be exercised before M1's orchestrator lands.
    Yields a single token event containing a placeholder answer.
    """

    def __init__(self, session: SessionState) -> None:  # noqa: ARG002
        self.session = session

    async def run_turn(
        self,
        user_message: str,
        approval_callback=None,
    ) -> AsyncIterator[Any]:
        yield _StubTokenEvent(
            f"[stub] Orchestrator not yet available. "
            f"You asked: "{user_message}". "
            f"Active RAG mode: {self.session.active_mode}."
        )
        yield _StubTurnCompleteEvent()


def _build_orchestrator(session: SessionState) -> Any:
    if _ORCHESTRATOR_AVAILABLE:
        return Orchestrator(session=session)
    return _StubOrchestrator(session=session)


# ── Contract loading ─────────────────────────────────────────────────────────


def _load_contract(path: str, idx: int) -> dict:
    """Return the contract dict at position `idx` inside a ContractNLI JSON file."""
    fpath = Path(path)
    if not fpath.exists():
        print(f"[ERROR] Contract file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with fpath.open(encoding="utf-8") as fh:
        data = json.load(fh)
    docs = data.get("documents") or data  # handle both wrapped and bare arrays
    if isinstance(docs, dict):
        docs = list(docs.values())
    if not isinstance(docs, list) or idx >= len(docs):
        print(
            f"[ERROR] Index {idx} out of range — file has {len(docs) if isinstance(docs, list) else '?'} documents.",
            file=sys.stderr,
        )
        sys.exit(1)
    return docs[idx]


# ── RunTrace persistence helpers ──────────────────────────────────────────────


def _persist_session(session: SessionState, recorder: RunTraceRecorder) -> Optional[Path]:
    """Write (or overwrite) the session RunTrace. Returns the path."""
    rt = build_session_runtrace(
        session_id=session.session_id,
        contract_id=session.contract_id,
        retrieval_mode=session.active_mode,
        started_at=session.started_at,
        conversation_history=session.conversation_history,
        tool_calls=recorder.snapshot(),
        approval_events=session.approval_events,
        retrieval_mode_switches=session.mode_switches,
        referenced_contract_runtraces=session.contract_runtraces,
    )
    return write_session_runtrace(rt)


def _finalise_session(
    session: SessionState,
    recorder: RunTraceRecorder,
    tui: TUI,
) -> None:
    rt = build_session_runtrace(
        session_id=session.session_id,
        contract_id=session.contract_id,
        retrieval_mode=session.active_mode,
        started_at=session.started_at,
        ended_at=utc_now_iso(),
        conversation_history=session.conversation_history,
        tool_calls=recorder.snapshot(),
        approval_events=session.approval_events,
        retrieval_mode_switches=session.mode_switches,
        referenced_contract_runtraces=session.contract_runtraces,
    )
    path = write_session_runtrace(rt)
    tui.print_goodbye(str(path))


# ── Orchestrator turn runner ──────────────────────────────────────────────────


async def _run_orchestrator_turn(
    orch: Any,
    user_message: str,
    tui: TUI,
    session: SessionState,
    recorder: RunTraceRecorder,
) -> tuple[str, Optional[str]]:
    """
    Drive one round-trip with the orchestrator.

    Handles:
      TokenEvent / ThinkEvent  → streamed to TUI
      StatusEvent              → status line in TUI
      ToolCallEvent            → tool-call JSON in dev mode
      ApprovalRequestEvent     → approval prompt; records ApprovalEvent
      TurnCompleteEvent        → citation block
    """
    answer_tokens: list[str] = []
    think_tokens: list[str] = []
    citations: dict = {"contract": [], "external": []}
    answer_started = False

    # Wrap the orchestrator's async generator as a sync token iterator
    # for tui.stream_response, or handle events directly.
    async for event in orch.run_turn(
        user_message,
        approval_callback=lambda tool, args, est=None: _handle_approval(
            tui, session, tool, args, est
        ),
    ):
        cls_name = type(event).__name__

        # Token / think (stub and real paths both handled here)
        if cls_name in ("TokenEvent", "_StubTokenEvent"):
            if not answer_started:
                tui.render_answer_header()
                answer_started = True
            text: str = event.text
            if getattr(event, "is_thinking", False):
                think_tokens.append(text)
                if session.dev_mode:
                    import sys as _sys
                    print(f"\033[2m{text}\033[0m" if _sys.stdout.isatty() else text, end="", flush=True)
            else:
                answer_tokens.append(text)
                print(text, end="", flush=True)

        elif cls_name == "ThinkEvent":
            think_tokens.append(event.text)
            if session.dev_mode:
                import sys as _sys
                print(f"\033[2m{event.text}\033[0m" if _sys.stdout.isatty() else event.text, end="", flush=True)

        elif cls_name == "StatusEvent":
            tui.render_status(event.message)

        elif cls_name == "ToolCallEvent":
            tui.render_tool_call_json(
                event.name,
                getattr(event, "arguments", {}),
                getattr(event, "output", None),
                getattr(event, "duration_ms", 0.0),
            )

        elif cls_name in ("TurnCompleteEvent", "_StubTurnCompleteEvent"):
            cites = getattr(event, "citations", {})
            citations = cites if isinstance(cites, dict) else {}

    if answer_started:
        print()  # newline after streamed answer

    tui.render_citations(
        citations.get("contract", []),
        citations.get("external", []),
    )

    return "".join(answer_tokens), ("".join(think_tokens) or None)


def _handle_approval(
    tui: TUI,
    session: SessionState,
    tool: str,
    args: dict,
    estimated_s: Optional[float],
) -> bool:
    approved = tui.render_approval_prompt(tool, args, estimated_s)
    event: ApprovalEvent = {
        "tool": tool,
        "arguments": args,
        "approved": approved,
        "approved_by": "user",
        "timestamp": utc_now_iso(),
    }
    session.record_approval(event)
    return approved


# ── REPL loop ─────────────────────────────────────────────────────────────────


async def _repl(
    args: argparse.Namespace,
    session: SessionState,
    recorder: RunTraceRecorder,
    tui: TUI,
) -> None:
    orch = _build_orchestrator(session)
    _persist_session(session, recorder)  # write initial snapshot

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        if raw.startswith("/"):
            result = tui.handle_slash_command(raw)
            if result == "exit":
                break
            if result == "analyze":
                # Synthesise a request that the orchestrator routes to run_full_analysis
                raw = f"Please run the full 17-hypothesis analysis on contract {session.contract_id}."
            else:
                _persist_session(session, recorder)
                continue

        session.add_turn("user", raw)
        answer, _ = await _run_orchestrator_turn(orch, raw, tui, session, recorder)
        session.add_turn("assistant", answer)
        _persist_session(session, recorder)

    _finalise_session(session, recorder, tui)


# ── One-shot mode ─────────────────────────────────────────────────────────────


async def _one_shot(
    args: argparse.Namespace,
    session: SessionState,
    recorder: RunTraceRecorder,
    tui: TUI,
) -> None:
    orch = _build_orchestrator(session)
    session.add_turn("user", args.prompt)
    answer, _ = await _run_orchestrator_turn(orch, args.prompt, tui, session, recorder)
    session.add_turn("assistant", answer)
    _finalise_session(session, recorder, tui)


# ── Eval mode ─────────────────────────────────────────────────────────────────


def _run_eval() -> None:
    try:
        from pipeline import eval_ms3  # type: ignore[import]
        asyncio.run(eval_ms3.run_batch())
    except ImportError:
        print(
            "[INFO] pipeline/eval_ms3.py not yet available — "
            "eval mode will be enabled once M5 lands the evaluation runner.",
            file=sys.stderr,
        )
        sys.exit(0)


# ── Argument parser ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="ContractLens MS3 — agentic NDA analysis REPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python agent.py --contract data/test.json --idx 0\n"
            "  python agent.py --contract data/test.json --idx 0 --dev\n"
            "  python agent.py --contract data/test.json --idx 0 \\\n"
            "                  --prompt \"Who are the parties?\"\n"
            "  python agent.py --eval"
        ),
    )
    parser.add_argument(
        "--contract",
        default="data/test.json",
        metavar="PATH",
        help="Path to ContractNLI JSON file (default: data/test.json)",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        metavar="N",
        help="Zero-based document index within the file (default: 0)",
    )
    parser.add_argument(
        "--dev", "--verbose",
        dest="dev",
        action="store_true",
        help="Boot in developer mode (shows <think> blocks and raw tool-call JSON)",
    )
    parser.add_argument(
        "--prompt",
        metavar="TEXT",
        help="One-shot question — answer and exit without entering the REPL",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Headless batch evaluation over all 123 test NDAs (M5's runner)",
    )
    return parser


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.eval:
        _run_eval()
        return

    contract = _load_contract(args.contract, args.idx)
    contract_id: str = contract.get("contract_id") or f"doc_{args.idx:03d}"

    session = SessionState.create(
        contract_id=contract_id,
        contract=contract,
        dev_mode=args.dev,
    )

    recorder = RunTraceRecorder()
    set_active_recorder(recorder)

    tui = TUI(session)
    tui.print_banner(args.contract, args.idx)

    if args.prompt:
        asyncio.run(_one_shot(args, session, recorder, tui))
    else:
        asyncio.run(_repl(args, session, recorder, tui))


if __name__ == "__main__":
    main()
