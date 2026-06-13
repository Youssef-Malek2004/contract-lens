"""
src/approval.py

Approval gate for heavy tools (currently only `run_full_analysis`).

The orchestrator's tool handler calls `gate.request(tool, arguments)` and
awaits a boolean. How that boolean is obtained — interactive prompt, GUI
dialog, auto-approve in --eval mode — is the gate's concern, not the
handler's.

Two ready-to-use prompts ship here:

  - `console_prompt`  — Claude-Code-style Y/n on stdin/stderr. Used by
    the headless demo + agent.py's user-mode REPL.
  - `auto_approve_prompt` — always returns True. Used by M5's --eval mode
    so the batch runner doesn't block waiting for human input.

M4 wires their TUI by passing a custom `prompt_fn` that pushes the request
onto the UI's event loop and returns a future the UI completes.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from textwrap import indent
from typing import Awaitable, Callable, List, Optional, Tuple

from src.types import ApprovalEvent


PromptFn = Callable[[str, dict], Awaitable[bool]]


@dataclass
class ApprovalGate:
    """
    Async approval primitive. Pass `.request` as the `approval_gate` on a
    ToolContext.

    `prompt_fn` is the user-facing prompt — it receives `(tool, arguments)`
    and returns an awaitable bool. If omitted, defaults to the console
    prompt; pass `auto_approve=True` to skip the prompt entirely.

    A small in-memory history of (tool, args, approved) tuples is kept for
    test / debug introspection.
    """
    prompt_fn: Optional[PromptFn] = None
    auto_approve: bool = False
    history: List[Tuple[str, dict, bool]] = field(default_factory=list)

    async def request(self, tool: str, arguments: dict) -> bool:
        if self.auto_approve:
            approved = True
        else:
            fn = self.prompt_fn or console_prompt
            approved = bool(await fn(tool, arguments))
        self.history.append((tool, dict(arguments), approved))
        return approved


# ── Built-in prompts ────────────────────────────────────────────────────────


async def console_prompt(tool: str, arguments: dict) -> bool:
    """
    Renders a Claude-Code-style approval prompt on stderr and reads stdin.

    Runs `input()` in an executor so the asyncio loop stays unblocked. The
    prompt goes to stderr so it doesn't interleave with streamed answer
    tokens on stdout.
    """
    args_blob = json.dumps(arguments, indent=2, ensure_ascii=False, default=str)
    body = (
        "\n"
        "  ⚠  Orchestrator wants to run a heavy tool:\n"
        f"     {tool}(\n"
        f"{indent(args_blob, '       ')}\n"
        "     )\n"
        "     Proceed? [Y/n] "
    )
    sys.stderr.write(body)
    sys.stderr.flush()

    loop = asyncio.get_running_loop()
    line = await loop.run_in_executor(None, sys.stdin.readline)
    answer = (line or "").strip().lower()
    return answer in ("", "y", "yes")


async def auto_approve_prompt(tool: str, arguments: dict) -> bool:
    return True


# ── Helpers ─────────────────────────────────────────────────────────────────


def to_event(
    tool: str,
    arguments: dict,
    approved: bool,
    timestamp: str,
) -> ApprovalEvent:
    """Build an ApprovalEvent dict — mirrors the schema expected by RunTrace."""
    return {
        "tool": tool,
        "arguments": dict(arguments),
        "approved": bool(approved),
        "approved_by": "user",
        "timestamp": timestamp,
    }
