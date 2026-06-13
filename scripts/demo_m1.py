#!/usr/bin/env python3
"""
scripts/demo_m1.py

Interactive REPL that exercises M1's slice end-to-end so we can see the
orchestrator thinking, choosing tools, dispatching them, and getting them
logged — before M2/M3/M4 land their pieces.

What's real:
  - Orchestrator (Qwen 3.5-27B on OpenRouter, SSE streamed)
  - Three tool handlers (retrieve, lookup_hypothesis, run_full_analysis)
  - Tool-call recorder + session-RunTrace writer
  - Approval gate (Claude-Code-style Y/n console prompt)
  - Vector / graph RAG via the MS2 retrievers (if indexes exist)

What's stubbed:
  - run_full_analysis pipeline — returns a fake 17-trace summary so we can
    observe the approval gate + cache + audit chain end-to-end without
    waiting on M2.

Usage:
    python scripts/demo_m1.py                       # NDA 0 from data/test.json
    python scripts/demo_m1.py --idx 5 --dev         # NDA 5 in developer mode
    python scripts/demo_m1.py --auto-approve        # skip the Y/n prompt
    python scripts/demo_m1.py --no-rag              # stub retrievers (no FAISS)

Slash commands:
    /dev /user           toggle reasoning display
    /vector-rag          switch session to FAISS vector RAG
    /graph-rag           switch session to networkx GraphRAG
    /analyze             force a run_full_analysis request
    /reset               clear conversation history
    /exit                quit (writes session RunTrace)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# Repo root on sys.path so `python scripts/demo_m1.py` works without -m.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.approval import ApprovalGate, console_prompt, auto_approve_prompt  # noqa: E402
from src.bootstrap import setup_runtime  # noqa: E402
from src.orchestrator import Orchestrator, OrchestratorEvent  # noqa: E402
from src.runtrace import (  # noqa: E402
    build_session_runtrace, make_session_id, utc_now_iso,
    write_session_runtrace,
)


# ── ANSI ───────────────────────────────────────────────────────────────────

DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
GREY   = "\033[2;37m"
RED    = "\033[31m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


# ── Demo session state (satisfies SessionLike) ─────────────────────────────


@dataclass
class DemoSession:
    contract_id: str
    session_id: str
    started_at: str
    retrieval_mode: str = "vector"
    cached_traces: Dict[str, Any] = field(default_factory=dict)
    approval_events: List[dict] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)            # ConversationTurn shape
    mode_switches: List[dict] = field(default_factory=list)
    referenced_runtraces: List[str] = field(default_factory=list)
    dev_mode: bool = False


# ── Stub pipeline (M2's responsibility — fake for now) ─────────────────────


async def stub_pipeline(contract_id: str, retrieval_mode: str) -> dict:
    """Pretend to run 17 hypothesis workers. Returns a fake summary."""
    await asyncio.sleep(0.4)  # simulate fan-out
    traces = []
    for i in range(1, 18):
        label = ["ENTAILED", "NOT_MENTIONED", "CONTRADICTED"][i % 3]
        traces.append({
            "hypothesis_id": f"H{i:02}",
            "label": label,
            "confidence": round(0.65 + 0.02 * i, 3),
            "evidence_spans": [],
            "verbatim_quote": None,
            "groundedness_check": True,
            "quote_integrity_check": True,
            "playbook_result": {
                "severity": "LOW",
                "action": "ACCEPT",
                "rationale": "stub-pipeline placeholder",
            },
            "agent_metadata": {
                "agent_id": f"H{i:02}", "rag_query": "stub",
                "rag_hits": 0, "rag_mode": retrieval_mode,
            },
        })
    label_counts = {"ENTAILED": 0, "CONTRADICTED": 0, "NOT_MENTIONED": 0}
    for t in traces:
        label_counts[t["label"]] += 1
    return {
        "status": "ok",
        "contract_id": contract_id,
        "retrieval_mode": retrieval_mode,
        "hypothesis_traces": traces,
        "summary": (
            f"Stub analysis complete on {contract_id} ({retrieval_mode}): "
            f"{label_counts['ENTAILED']} ENTAILED, "
            f"{label_counts['CONTRADICTED']} CONTRADICTED, "
            f"{label_counts['NOT_MENTIONED']} NOT_MENTIONED."
        ),
        "metrics": {"hypothesis_count": 17, "label_counts": label_counts},
    }


# ── Stub retrievers (fallback when no FAISS / index missing) ──────────────


@dataclass
class StubRetriever:
    label: str
    def retrieve(self, query, top_k=5, hypothesis_id=None, label_filter=None):
        return [{
            "text": f"[stub-{self.label}] example span echoing {query!r}",
            "doc_id": "doc_stub", "span_idx": 0, "score": 0.5,
            "hypothesis_annotations": {hypothesis_id: label_filter} if hypothesis_id and label_filter else {},
        }]


# ── Contract loading ──────────────────────────────────────────────────────


def load_contract(path: Path, idx: int, max_chars: int = 12000) -> tuple[str, str, str]:
    """Returns (contract_id, file_name, truncated_text)."""
    data = json.loads(path.read_text())
    docs = data["documents"] if isinstance(data, dict) else data
    if idx < 0 or idx >= len(docs):
        raise SystemExit(f"--idx {idx} out of range (0..{len(docs)-1})")
    doc = docs[idx]
    text = doc.get("text", "")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[... contract truncated at {max_chars:,} chars for demo ...]"
    return f"doc_{doc.get('id', idx)}", doc.get("file_name", "unknown"), text


# ── Event renderer ────────────────────────────────────────────────────────


class TurnRenderer:
    """
    Renders an OrchestratorEvent stream to the terminal.

    In user mode: hide think tokens, show a single 'reasoning…' status while
    they stream. In dev mode: stream reasoning inline in dim grey.
    """
    def __init__(self, dev: bool):
        self.dev = dev
        self._content_started = False
        self._reasoning_active = False
        self._answer_chars: list[str] = []

    def handle(self, ev: OrchestratorEvent) -> None:
        if ev.kind == "think" and ev.text:
            if self.dev:
                if not self._reasoning_active:
                    sys.stdout.write(_c("\n  <think> ", GREY))
                    self._reasoning_active = True
                sys.stdout.write(_c(ev.text, GREY))
                sys.stdout.flush()
            else:
                if not self._reasoning_active:
                    sys.stdout.write(_c("  reasoning…", DIM))
                    sys.stdout.flush()
                    self._reasoning_active = True

        elif ev.kind == "content" and ev.text:
            if self._reasoning_active:
                if self.dev:
                    sys.stdout.write(_c(" </think>\n\n", GREY))
                else:
                    sys.stdout.write("\r" + " " * 20 + "\r")  # erase 'reasoning…'
                self._reasoning_active = False
            if not self._content_started:
                self._content_started = True
            sys.stdout.write(ev.text)
            sys.stdout.flush()
            self._answer_chars.append(ev.text)

        elif ev.kind == "tool_call":
            if self._reasoning_active:
                sys.stdout.write(_c(" </think>\n", GREY) if self.dev else "\r" + " " * 20 + "\r")
                self._reasoning_active = False
            args_repr = json.dumps(ev.tool_arguments or {}, ensure_ascii=False)
            if len(args_repr) > 120:
                args_repr = args_repr[:117] + "..."
            sys.stdout.write("\n" + _c(f"  ⤷ tool   {ev.tool_name}({args_repr})", CYAN) + "\n")
            sys.stdout.flush()

        elif ev.kind == "tool_result":
            summary = _summarize_output(ev.tool_output)
            sys.stdout.write(_c(f"  ⤶ result {ev.tool_name} → {summary}", DIM + CYAN) + "\n")
            sys.stdout.flush()

        elif ev.kind == "turn_complete":
            if self._reasoning_active and self.dev:
                sys.stdout.write(_c(" </think>", GREY))
            sys.stdout.write("\n")
            sys.stdout.flush()

    @property
    def answer_text(self) -> str:
        return "".join(self._answer_chars)


def _summarize_output(output: Any) -> str:
    if output is None:
        return "None"
    if isinstance(output, list):
        return f"list[{len(output)}]" + (f" first={output[0]!r}"[:60] if output else "")
    if isinstance(output, dict):
        keys = ", ".join(list(output.keys())[:5])
        s = json.dumps(output, default=str, ensure_ascii=False)
        return f"{{{keys}}} ({len(s)} chars)"
    return repr(output)[:120]


# ── REPL ──────────────────────────────────────────────────────────────────


SLASH_HELP = """\
slash commands:
  /dev            verbose mode — stream <think> reasoning
  /user           default mode — hide reasoning
  /vector-rag     switch session retrieval to FAISS vector RAG
  /graph-rag      switch session retrieval to networkx GraphRAG
  /analyze        force the full 17-hypothesis run (still goes through approval)
  /reset          clear conversation history
  /audit          print all recorded tool calls so far
  /exit           quit and write session RunTrace
"""


def _to_orchestrator_history(turns: List[dict]) -> List[dict]:
    """Strip timestamps / citations — return plain {role, content} for the model."""
    return [{"role": t["role"], "content": t["content"]} for t in turns]


def _handle_slash(cmd: str, session: DemoSession, recorder) -> tuple[bool, str | None]:
    """
    Returns (handled, follow_up_prompt). When `follow_up_prompt` is non-None,
    the REPL feeds it to the orchestrator as a synthetic user message
    (used by /analyze).
    """
    cmd = cmd.strip().lower()
    if cmd == "/exit":
        return True, "__EXIT__"
    if cmd in ("/dev", "/verbose"):
        session.dev_mode = True
        print(_c("  → dev mode ON (showing <think>)", YELLOW))
        return True, None
    if cmd == "/user":
        session.dev_mode = False
        print(_c("  → user mode (hiding <think>)", YELLOW))
        return True, None
    if cmd in ("/vector-rag", "/graph-rag"):
        new_mode = "vector" if cmd == "/vector-rag" else "graph"
        if new_mode != session.retrieval_mode:
            session.mode_switches.append({
                "from_mode": session.retrieval_mode,
                "to_mode": new_mode,
                "timestamp": utc_now_iso(),
            })
            session.retrieval_mode = new_mode
            print(_c(f"  → retrieval mode switched to {new_mode}", YELLOW))
        else:
            print(_c(f"  → already in {new_mode} mode", DIM))
        return True, None
    if cmd == "/reset":
        # Keep the system "contract" message at index 0; drop everything else.
        session.history = session.history[:1]
        print(_c("  → conversation history cleared (contract retained)", YELLOW))
        return True, None
    if cmd == "/audit":
        for e in recorder.snapshot():
            print(_c(
                f"  • {e['agent_id']:13} {e['name']:18} "
                f"count={e['count']} dur={e['duration_ms']:.0f}ms "
                f"args={json.dumps(e.get('arguments') or {}, default=str)[:80]}",
                DIM,
            ))
        return True, None
    if cmd == "/analyze":
        return True, "Please run the full 17-hypothesis analysis on this contract."
    if cmd in ("/help", "/?"):
        print(SLASH_HELP)
        return True, None
    print(_c(f"  ? unknown slash command: {cmd}", RED))
    return True, None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--contract", default="data/test.json")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--dev", action="store_true", help="boot in dev mode (show <think>)")
    parser.add_argument("--auto-approve", action="store_true",
                        help="skip the run_full_analysis Y/n prompt")
    parser.add_argument("--no-rag", action="store_true",
                        help="use stub retrievers instead of FAISS / networkx")
    parser.add_argument("--prompt", default=None,
                        help="one-shot: run a single prompt and exit (no REPL)")
    args = parser.parse_args()

    # 1. Load the contract
    contract_id, file_name, text = load_contract(Path(args.contract), args.idx)
    print(_c(f"\n=== ContractLens M1 demo ===", BOLD))
    print(_c(f"contract: {contract_id}  ({file_name})  {len(text):,} chars", DIM))

    # 2. Build the session
    started = utc_now_iso()
    session = DemoSession(
        contract_id=contract_id,
        session_id=make_session_id(contract_id),
        started_at=started,
        dev_mode=args.dev,
    )
    # Inject the contract as the first system message so it travels with every turn.
    session.history.append({
        "role": "system",
        "content": f"[CONTRACT id={contract_id}]\n{text}\n[/CONTRACT]",
        "timestamp": started,
    })

    # 3. Approval gate
    gate = ApprovalGate(
        prompt_fn=(auto_approve_prompt if args.auto_approve else console_prompt),
        auto_approve=args.auto_approve,
    )

    # 4. Bind tool context
    if args.no_rag:
        rag_v = StubRetriever("vector"); rag_g = StubRetriever("graph")
        recorder = setup_runtime(
            session=session, pipeline=stub_pipeline, approval_gate=gate.request,
            rag_vector=rag_v, rag_graph=rag_g, use_real_rag=False,
        )
        print(_c("rag: STUB (use real indexes by dropping --no-rag)", DIM))
    else:
        recorder = setup_runtime(
            session=session, pipeline=stub_pipeline, approval_gate=gate.request,
            use_real_rag=True,
        )
        print(_c("rag: REAL (FAISS + networkx) — pipeline is STUB", DIM))

    # 5. Orchestrator
    try:
        orch = Orchestrator()
    except RuntimeError as e:
        print(_c(f"\nERROR: {e}", RED))
        return 2
    print(_c(f"orchestrator: {orch.model_id}", DIM))
    print(_c(f"session_id: {session.session_id}", DIM))
    print(_c("type a question, /help, or /exit\n", DIM))

    # 6. REPL
    async def one_turn(user_input: str) -> None:
        ts = utc_now_iso()
        session.history.append({"role": "user", "content": user_input, "timestamp": ts})
        renderer = TurnRenderer(dev=session.dev_mode)

        try:
            async for ev in orch.run_turn(
                user_message=user_input,
                history=_to_orchestrator_history(session.history[:-1]),  # everything before this user msg
                contract_block=None,  # contract already in history as system msg
            ):
                renderer.handle(ev)
        except Exception as e:
            print(_c(f"\n  orchestrator error: {e!r}", RED))
            return

        session.history.append({
            "role": "assistant",
            "content": renderer.answer_text,
            "timestamp": utc_now_iso(),
        })
        # Snapshot session RunTrace after every turn (atomic write).
        _flush_session_runtrace(session, recorder)

    if args.prompt:
        await one_turn(args.prompt)
        _flush_session_runtrace(session, recorder, final=True)
        return 0

    while True:
        try:
            user_input = input(_c("\n> ", GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.startswith("/"):
            handled, follow_up = _handle_slash(user_input, session, recorder)
            if follow_up == "__EXIT__":
                break
            if follow_up:
                await one_turn(follow_up)
            continue
        await one_turn(user_input)

    _flush_session_runtrace(session, recorder, final=True)
    return 0


def _flush_session_runtrace(session: DemoSession, recorder, final: bool = False) -> None:
    rt = build_session_runtrace(
        session_id=session.session_id,
        contract_id=session.contract_id,
        retrieval_mode=session.retrieval_mode,
        started_at=session.started_at,
        ended_at=utc_now_iso(),
        conversation_history=session.history,
        tool_calls=recorder.snapshot(),
        approval_events=session.approval_events,
        retrieval_mode_switches=session.mode_switches,
        referenced_contract_runtraces=session.referenced_runtraces,
    )
    path = write_session_runtrace(rt)
    if final:
        print(_c(f"\nsession runtrace → {path}", DIM))
        print(_c(
            f"recorded: {len(recorder.snapshot())} tool calls, "
            f"{len(session.approval_events)} approval events, "
            f"{len(session.mode_switches)} mode switches",
            DIM,
        ))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
