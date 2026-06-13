#!/usr/bin/env python3
"""
agent.py — ContractLens MS3 REPL entry point.

Usage:
    python agent.py --contract data/test.json --idx 0           # REPL, user mode
    python agent.py --contract data/test.json --idx 0 --dev     # REPL, developer mode
    python agent.py --contract data/test.json --idx 0 \\
                    --prompt "Who are the parties?"              # one-shot, no REPL
    python agent.py --eval                                       # headless batch (M5)
    python agent.py --contract data/test.json --idx 0 --no-rag  # skip FAISS (stub retrievers)

Environment:
    OPENROUTER_API_KEY — required for live inference (set in .env)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.approval import ApprovalGate, auto_approve_prompt
from src.bootstrap import setup_runtime
from src.orchestrator import Orchestrator
from src.runtrace import (
    build_session_runtrace,
    utc_now_iso,
    write_session_runtrace,
)
from src.dispatcher import make_pipeline
from src.session import SessionState
from src.tui import TUI


# ── Contract loading ──────────────────────────────────────────────────────────

_MAX_CONTRACT_CHARS = 12_000


def _load_contract(path: str, idx: int) -> tuple:
    """
    Returns (doc_dict, contract_id, contract_text).

    contract_text is truncated to _MAX_CONTRACT_CHARS to fit the context
    window — mirrors demo_m1.py's behaviour.
    """
    fpath = Path(path)
    if not fpath.exists():
        print(f"[ERROR] Contract file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with fpath.open(encoding="utf-8") as fh:
        data = json.load(fh)

    docs = data["documents"] if isinstance(data, dict) else data
    if not isinstance(docs, list) or idx < 0 or idx >= len(docs):
        count = len(docs) if isinstance(docs, list) else "?"
        print(f"[ERROR] Index {idx} out of range — file has {count} documents.", file=sys.stderr)
        sys.exit(1)

    doc = docs[idx]
    contract_id = f"doc_{doc.get('id', idx)}"
    text: str = doc.get("text", "")
    if len(text) > _MAX_CONTRACT_CHARS:
        text = text[:_MAX_CONTRACT_CHARS] + f"\n\n[... contract truncated at {_MAX_CONTRACT_CHARS:,} chars ...]"
    return doc, contract_id, text


# ── History helpers ───────────────────────────────────────────────────────────


def _to_orchestrator_history(turns: list) -> List[dict]:
    """Strip timestamps/citations — return plain {role, content} for the model."""
    return [{"role": t["role"], "content": t["content"]} for t in turns]


# ── RunTrace persistence ──────────────────────────────────────────────────────


def _persist_session(session: SessionState, recorder, ended: bool = False) -> Path:
    rt = build_session_runtrace(
        session_id=session.session_id,
        contract_id=session.contract_id,
        retrieval_mode=session.retrieval_mode,
        started_at=session.started_at,
        ended_at=utc_now_iso() if ended else None,
        conversation_history=session.conversation_history,
        tool_calls=recorder.snapshot(),
        approval_events=session.approval_events,
        retrieval_mode_switches=session.mode_switches,
        referenced_contract_runtraces=session.contract_runtraces,
    )
    # Backward-compat with MS2: mirror conversation history to a flat JSON file.
    _write_conversation_history(session)
    return write_session_runtrace(rt)


def _write_conversation_history(session: SessionState) -> None:
    path = Path("conversation_history.json")
    turns = [{"role": t["role"], "content": t["content"]} for t in session.conversation_history]
    fd, tmp = tempfile.mkstemp(prefix="conv_hist.", dir=".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(turns, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


# ── One turn ─────────────────────────────────────────────────────────────────


async def _one_turn(
    orch: Orchestrator,
    user_message: str,
    session: SessionState,
    tui: TUI,
) -> str:
    """
    Drive one round-trip with the orchestrator and render events to the TUI.
    Returns the assistant's answer text.
    """
    _IS_TTY = sys.stdout.isatty()

    def _c(text: str, code: str) -> str:
        return f"{code}{text}\033[0m" if _IS_TTY else text

    # Pass history up to (but not including) the new user message.
    history = _to_orchestrator_history(session.conversation_history)

    answer_parts: list[str] = []
    content_started = False
    reasoning_active = False
    _last_tool_args: dict | None = None

    try:
        async for ev in orch.run_turn(
            user_message=user_message,
            history=history,
            contract_block=None,  # contract already in history as system message
        ):
            if ev.kind == "think" and ev.text:
                if session.dev_mode:
                    if not reasoning_active:
                        sys.stdout.write(_c("\n  <think> ", "\033[2;37m"))
                        reasoning_active = True
                    sys.stdout.write(_c(ev.text, "\033[2;37m"))
                    sys.stdout.flush()
                else:
                    if not reasoning_active:
                        sys.stdout.write(_c("  reasoning…", "\033[2m"))
                        sys.stdout.flush()
                        reasoning_active = True

            elif ev.kind == "content" and ev.text:
                if reasoning_active:
                    if session.dev_mode:
                        sys.stdout.write(_c(" </think>\n\n", "\033[2;37m"))
                    else:
                        sys.stdout.write("\r" + " " * 20 + "\r")
                    reasoning_active = False
                if not content_started:
                    tui.render_answer_header()
                    content_started = True
                sys.stdout.write(ev.text)
                sys.stdout.flush()
                answer_parts.append(ev.text)

            elif ev.kind == "tool_call":
                if reasoning_active:
                    sys.stdout.write("\n")
                    reasoning_active = False
                _last_tool_args = ev.tool_arguments or {}
                args_repr = json.dumps(_last_tool_args, ensure_ascii=False)
                if len(args_repr) > 120:
                    args_repr = args_repr[:117] + "..."
                sys.stdout.write("\n" + _c(f"  ⤷ tool   {ev.tool_name}({args_repr})", "\033[36m") + "\n")
                sys.stdout.flush()

            elif ev.kind == "tool_result":
                summary = _summarize(ev.tool_output)
                sys.stdout.write(_c(f"  ⤶ result {ev.tool_name} → {summary}", "\033[2;36m") + "\n")
                sys.stdout.flush()
                # Dev mode: full raw JSON payload (plan §2.9)
                if session.dev_mode and _last_tool_args is not None:
                    tui.render_tool_call_json(ev.tool_name, _last_tool_args, ev.tool_output, 0.0)
                    _last_tool_args = None
                # Track the contract RunTrace produced by run_full_analysis.
                if ev.tool_name == "run_full_analysis" and isinstance(ev.tool_output, dict):
                    runtrace_path = ev.tool_output.get("runtrace_path")
                    if runtrace_path:
                        session.add_contract_runtrace_ref(str(runtrace_path))

            elif ev.kind == "turn_complete":
                if reasoning_active and session.dev_mode:
                    sys.stdout.write(_c(" </think>", "\033[2;37m"))
                sys.stdout.write("\n")
                sys.stdout.flush()

    except Exception as exc:
        print(_c(f"\n  orchestrator error: {exc!r}", "\033[31m"))

    return "".join(answer_parts)


def _summarize(output) -> str:
    if output is None:
        return "None"
    if isinstance(output, list):
        return f"list[{len(output)}]"
    if isinstance(output, dict):
        keys = ", ".join(list(output.keys())[:4])
        return f"{{{keys}}} ({len(json.dumps(output, default=str))} chars)"
    return repr(output)[:100]


# ── REPL loop ─────────────────────────────────────────────────────────────────


async def _repl(
    args: argparse.Namespace,
    session: SessionState,
    recorder,
    tui: TUI,
) -> None:
    try:
        orch = Orchestrator()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    _persist_session(session, recorder)  # initial snapshot

    _IS_TTY = sys.stdout.isatty()
    prompt = f"\033[32m\n> \033[0m" if _IS_TTY else "\n> "

    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        if raw.startswith("/"):
            result = tui.handle_slash_command(raw, recorder=recorder)
            if result == "exit":
                break
            if result == "analyze":
                raw = "Please run the full 17-hypothesis analysis on this contract."
            else:
                _persist_session(session, recorder)
                continue

        session.add_turn("user", raw)
        answer = await _one_turn(orch, raw, session, tui)
        session.add_turn("assistant", answer)
        _persist_session(session, recorder)

    path = _persist_session(session, recorder, ended=True)
    tui.print_goodbye(str(path))
    _print_summary(session, recorder)


def _print_summary(session: SessionState, recorder) -> None:
    _IS_TTY = sys.stdout.isatty()
    dim = "\033[2m" if _IS_TTY else ""
    reset = "\033[0m" if _IS_TTY else ""
    calls = recorder.snapshot()
    print(
        f"{dim}  recorded: {len(calls)} tool calls, "
        f"{len(session.approval_events)} approval events, "
        f"{len(session.mode_switches)} mode switches{reset}"
    )


# ── One-shot mode ─────────────────────────────────────────────────────────────


async def _one_shot(
    args: argparse.Namespace,
    session: SessionState,
    recorder,
    tui: TUI,
) -> None:
    try:
        orch = Orchestrator()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    session.add_turn("user", args.prompt)
    answer = await _one_turn(orch, args.prompt, session, tui)
    session.add_turn("assistant", answer)
    path = _persist_session(session, recorder, ended=True)
    tui.print_goodbye(str(path))


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
    parser.add_argument("--contract", default="data/test.json", metavar="PATH",
                        help="Path to ContractNLI JSON file (default: data/test.json)")
    parser.add_argument("--idx", type=int, default=0, metavar="N",
                        help="Zero-based document index within the file (default: 0)")
    parser.add_argument("--dev", "--verbose", dest="dev", action="store_true",
                        help="Boot in developer mode (stream <think> reasoning + raw tool JSON)")
    parser.add_argument("--prompt", metavar="TEXT",
                        help="One-shot question — answer and exit without entering the REPL")
    parser.add_argument("--eval", action="store_true",
                        help="Headless batch evaluation over all 123 test NDAs (M5's runner)")
    parser.add_argument("--no-rag", dest="no_rag", action="store_true",
                        help="Use stub retrievers (skip FAISS / networkx — useful when indexes not built)")
    parser.add_argument("--auto-approve", dest="auto_approve", action="store_true",
                        help="Skip run_full_analysis approval prompt (for scripted / eval use)")
    return parser


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.eval:
        _run_eval()
        return

    doc, contract_id, contract_text = _load_contract(args.contract, args.idx)

    session = SessionState.create(
        contract_id=contract_id,
        contract=doc,
        dev_mode=args.dev,
    )

    # Inject the contract text as the first conversation entry so every
    # orchestrator turn has it in its context (same pattern as demo_m1.py).
    session.conversation_history.append({
        "role": "system",
        "content": f"[CONTRACT id={contract_id}]\n{contract_text}\n[/CONTRACT]",
        "timestamp": session.started_at,
    })

    # Approval gate — TUI prompt unless --auto-approve is set.
    gate = ApprovalGate(auto_approve=args.auto_approve)

    # Wire tool context + recorder via bootstrap helper.
    analysis_pipeline = make_pipeline(contract=doc, session_id=session.session_id)
    recorder = setup_runtime(
        session=session,
        pipeline=analysis_pipeline,           # M2's run_full_analysis — wired once M2 lands
        approval_gate=gate.request,
        use_real_rag=not args.no_rag,
    )

    tui = TUI(session)

    # Point the gate at the TUI's async approval renderer (after tui is built).
    if not args.auto_approve:
        gate.prompt_fn = tui.render_approval_prompt

    tui.print_banner(args.contract, args.idx)

    if args.prompt:
        asyncio.run(_one_shot(args, session, recorder, tui))
    else:
        asyncio.run(_repl(args, session, recorder, tui))


if __name__ == "__main__":
    main()
