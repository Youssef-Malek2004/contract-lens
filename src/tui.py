"""
src/tui.py

TUI renderer for the ContractLens REPL.

Two display modes (toggled via slash commands or --dev flag):

  User mode   — status icons, streamed answer, citation block, approval prompts
  Dev mode    — everything above plus <think> reasoning, raw tool-call JSON,
                per-hypothesis worker output, and per-stage timings

All output goes to stdout via plain print(). No third-party TUI library is
required — ANSI escapes are used only for bold/dim styling and are suppressed
when stdout is not a TTY.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Iterator, Optional, Tuple

from src.session import SessionState


# ── ANSI helpers ─────────────────────────────────────────────────────────────

_IS_TTY = sys.stdout.isatty()


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _IS_TTY else s


def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m" if _IS_TTY else s


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if _IS_TTY else s


def _cyan(s: str) -> str:
    return f"\033[36m{s}\033[0m" if _IS_TTY else s


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if _IS_TTY else s


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if _IS_TTY else s


# ── Think-tag stream parser ───────────────────────────────────────────────────


class _ThinkParser:
    """
    Stateful parser that splits a raw token stream into visible text and
    <think>…</think> reasoning blocks.

    Yields (is_thinking: bool, text: str) pairs. Handles partial tags that
    span chunk boundaries (e.g. chunk ends mid-way through "<thi").
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, chunk: str) -> Iterator[Tuple[bool, str]]:
        self._buf += chunk
        while True:
            if self._in_think:
                idx = self._buf.find(self._CLOSE)
                if idx == -1:
                    # Keep a tail in the buffer in case </think> spans chunks.
                    safe = max(0, len(self._buf) - len(self._CLOSE) + 1)
                    if safe:
                        yield (True, self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break
                yield (True, self._buf[:idx])
                self._buf = self._buf[idx + len(self._CLOSE):]
                self._in_think = False
            else:
                idx = self._buf.find(self._OPEN)
                if idx == -1:
                    # Keep a tail in case <think> spans chunks.
                    safe = max(0, len(self._buf) - len(self._OPEN) + 1)
                    if safe:
                        yield (False, self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break
                if idx > 0:
                    yield (False, self._buf[:idx])
                self._buf = self._buf[idx + len(self._OPEN):]
                self._in_think = True

    def flush(self) -> Iterator[Tuple[bool, str]]:
        """Drain any remaining buffer at stream end."""
        if self._buf:
            yield (self._in_think, self._buf)
            self._buf = ""
            self._in_think = False


# ── TUI ──────────────────────────────────────────────────────────────────────

_DIVIDER = "─" * 60
_THIN    = "─" * 40

_SLASH_COMMANDS = [
    ("/dev, /verbose", "Switch to developer mode"),
    ("/user",          "Switch back to user mode"),
    ("/vector-rag",    "Use FAISS vector RAG (default)"),
    ("/graph-rag",     "Use networkx GraphRAG"),
    ("/analyze",       "Force-trigger run_full_analysis (approval required)"),
    ("/reset",         "Clear conversation history for this contract"),
    ("/exit",          "Quit and save session RunTrace"),
    ("/help",          "Show this help"),
]


class TUI:
    """All rendering for one REPL session."""

    def __init__(self, session: SessionState) -> None:
        self.session = session

    # ── Banner / Help ─────────────────────────────────────────────────────────

    def print_banner(self, contract_path: str, idx: int) -> None:
        print(_DIVIDER)
        print(_bold("ContractLens  —  Conversation Agent  (MS3)"))
        print(_DIVIDER)
        print(f"  Contract  : {contract_path}  (idx={idx})")
        print(f"  ID        : {self.session.contract_id}")
        print(f"  Session   : {self.session.session_id}")
        print(f"  RAG mode  : {self.session.active_mode}")
        print(f"  Display   : {'developer' if self.session.dev_mode else 'user'}")
        print(_DIVIDER)
        print(_dim("Type a question or /help for commands."))
        print()

    def print_help(self) -> None:
        print()
        print(_bold("Slash commands"))
        print(_THIN)
        for cmd, desc in _SLASH_COMMANDS:
            print(f"  {_cyan(cmd):<22}  {desc}")
        print()

    # ── Status / progress ─────────────────────────────────────────────────────

    def render_status(self, msg: str, icon: str = "…") -> None:
        print(f"  {_dim(icon)}  {_dim(msg)}")

    def render_hypothesis_progress(
        self,
        h_id: str,
        label: str,
        latency_s: float,
        ok: bool = True,
    ) -> None:
        tick = _green("✓") if ok else _red("✗")
        label_str = _bold(label)
        print(f"    [{h_id}] {tick} {label_str} ({latency_s:.1f}s)")

    # ── Answer rendering ──────────────────────────────────────────────────────

    def render_answer_header(self) -> None:
        print()
        print(_bold("ANSWER"))
        print(_THIN)

    def stream_response(
        self,
        token_iter: Iterator[str],
    ) -> Tuple[str, Optional[str]]:
        """
        Consume a raw token stream, rendering tokens live.

        Returns (full_answer, think_content):
          - full_answer   : the non-think text concatenated
          - think_content : the content inside <think>…</think>, or None

        In user mode:  think blocks are hidden; only answer text is printed.
        In dev mode:   think blocks are printed in dim style before the answer.
        """
        parser = _ThinkParser()
        answer_parts: list[str] = []
        think_parts:  list[str] = []
        think_started = False

        for chunk in token_iter:
            for is_thinking, text in parser.feed(chunk):
                if is_thinking:
                    think_parts.append(text)
                    if self.session.dev_mode:
                        if not think_started:
                            print(_dim("  <think>"), flush=True)
                            think_started = True
                        print(_dim(text), end="", flush=True)
                else:
                    if think_started and self.session.dev_mode:
                        print(_dim("  </think>"))
                        print()
                        think_started = False
                    answer_parts.append(text)
                    print(text, end="", flush=True)

        for is_thinking, text in parser.flush():
            if is_thinking:
                think_parts.append(text)
                if self.session.dev_mode:
                    print(_dim(text), end="", flush=True)
            else:
                answer_parts.append(text)
                print(text, end="", flush=True)

        if think_started and self.session.dev_mode:
            print(_dim("  </think>"))

        print()  # newline after streamed answer
        return "".join(answer_parts), ("".join(think_parts) or None)

    # ── Citations ─────────────────────────────────────────────────────────────

    def render_citations(
        self,
        contract_cites: list,
        external_cites: list,
    ) -> None:
        if not contract_cites and not external_cites:
            return
        print()
        print(_bold("Sources"))
        if contract_cites:
            spans = "  ".join(str(c) for c in contract_cites)
            print(f"  {_cyan('[contract]')}  {spans}")
        if external_cites:
            for cite in external_cites:
                print(f"  {_dim('[external]')}  {cite}")

    # ── Developer-mode extras ─────────────────────────────────────────────────

    def render_tool_call_json(
        self,
        name: str,
        args: dict,
        output: Any,
        duration_ms: float,
    ) -> None:
        if not self.session.dev_mode:
            return
        payload = {
            "name": name,
            "arguments": args,
            "output": output,
            "duration_ms": round(duration_ms, 1),
        }
        print()
        print(_dim(f"  [tool_call] {name}"))
        print(_dim("  " + json.dumps(payload, indent=2, default=str).replace("\n", "\n  ")))

    def render_think_block(self, content: str) -> None:
        """Explicitly render a think block (used when dev mode is toggled on mid-stream)."""
        if not self.session.dev_mode:
            return
        print(_dim(f"  <think>{content}</think>"))

    # ── Approval prompt ───────────────────────────────────────────────────────

    def render_approval_prompt(
        self,
        tool: str,
        args: dict,
        estimated_s: Optional[float] = None,
    ) -> bool:
        """
        Render a Claude-Code-style approval gate and block on Y/N.
        Returns True if the user approves, False otherwise.
        Always visible in both user and dev mode.
        """
        time_hint = f" (~{estimated_s:.0f}s total)" if estimated_s else ""
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print()
        print(f"  {_yellow('⚠')}  Orchestrator wants to run: {_bold(tool)}({args_str})")
        if tool == "run_full_analysis":
            print(f"     This will perform 17 LLM calls{time_hint} and write a RunTrace.")
        print(f"     Proceed? [Y/n] ", end="", flush=True)
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        approved = answer in ("", "y", "yes")
        if not approved:
            print(_dim("  Skipped."))
        print()
        return approved

    # ── Slash command handler ─────────────────────────────────────────────────

    def handle_slash_command(self, raw: str) -> Optional[str]:
        """
        Process a slash command.

        Returns:
          "exit"    — caller should break the REPL loop
          "analyze" — caller should inject a synthetic run_full_analysis turn
          None      — command handled; continue the loop
        """
        cmd = raw.strip().lower()

        if cmd in ("/dev", "/verbose"):
            self.session.dev_mode = True
            print(_dim("  Switched to developer mode."))
            return None

        if cmd == "/user":
            self.session.dev_mode = False
            print(_dim("  Switched to user mode."))
            return None

        if cmd == "/vector-rag":
            self.session.switch_rag_mode("vector")
            print(_dim(f"  RAG mode → vector (FAISS)."))
            return None

        if cmd == "/graph-rag":
            self.session.switch_rag_mode("graph")
            print(_dim(f"  RAG mode → graph (networkx)."))
            return None

        if cmd == "/analyze":
            return "analyze"

        if cmd == "/reset":
            self.session.reset_history()
            print(_dim("  Conversation history cleared."))
            return None

        if cmd == "/exit":
            return "exit"

        if cmd == "/help":
            self.print_help()
            return None

        print(_dim(f"  Unknown command: {raw}  (type /help for the list)"))
        return None

    # ── Farewell ──────────────────────────────────────────────────────────────

    def print_goodbye(self, runtrace_path: str) -> None:
        print()
        print(_DIVIDER)
        print(_dim(f"  Session saved → {runtrace_path}"))
        print(_dim("  Goodbye."))
        print(_DIVIDER)
