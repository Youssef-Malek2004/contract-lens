#!/usr/bin/env python3
"""
test_model_loader.py
Smoke-test all three models through LocalLoader with live token streaming.
Thinking tokens stream in dim colour; response tokens stream normally.

Usage:
    conda activate genai-ms2
    python tests/test_model_loader.py
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import LocalLoader, get_device

PROMPT_ORCHESTRATOR = "What is a Non-Disclosure Agreement?"
PROMPT_BASE = "Summarise in one sentence: a receiving party must keep all shared information confidential."
PROMPT_NLI = (
    "Contract Spans:\n"
    '[0] "The Receiving Party shall not use Confidential Information for any '
    'purpose other than evaluation."\n\n'
    "Hypotheses:\n"
    "H04: Receiving Party shall not use any Confidential Information for any "
    "purpose other than the purposes stated in Agreement.\n\n"
    "Return ONLY a JSON array. No other text."
)

DIM   = "\033[2m"
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[32m"
RED   = "\033[31m"


def stream_and_print(handle, messages: list, max_new_tokens: int, enable_thinking: bool) -> str:
    """Stream from a ModelHandle, print chunks live with dim thinking, return full output."""
    stream = handle.stream(messages, max_new_tokens, enable_thinking=enable_thinking)
    full = ""
    in_think = False
    for chunk in stream:
        full += chunk
        if "<think>" in chunk:
            in_think = True
            print(DIM, end="", flush=True)
        if "</think>" in chunk:
            in_think = False
            print(chunk + RESET, end="", flush=True)
            continue
        print(chunk, end="", flush=True)
    if in_think:
        print(RESET, end="", flush=True)
    print()
    return full


def banner(title: str) -> None:
    print(f"\n{'='*55}\n  {BOLD}{title}{RESET}\n{'='*55}")


def result(passed: bool) -> None:
    print(f"{GREEN}  PASS{RESET}" if passed else f"{RED}  FAIL{RESET}")


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {BOLD}{device}{RESET}")
    loader = LocalLoader(device=device)

    # ── Test 1: Orchestrator (Qwen3-4B, thinking ON) ───────────────────────────
    banner("Test 1 — Orchestrator  (Qwen3-4B · thinking=ON)")
    print("Loading…", flush=True)
    t0 = time.time()
    orch = loader.load_orchestrator()
    print(f"Loaded {orch.model_id} in {time.time()-t0:.1f}s\n")
    print(f"Prompt: {PROMPT_ORCHESTRATOR}\n")
    t0 = time.time()
    reply = stream_and_print(orch, [{"role": "user", "content": PROMPT_ORCHESTRATOR}], 200, True)
    print(f"  [{time.time()-t0:.1f}s]")
    result(bool(reply.strip()))

    # ── Test 2: Base model (Qwen3-1.7B, thinking ON) ──────────────────────────
    banner("Test 2 — Base model  (Qwen3-1.7B · thinking=ON)")
    print("Loading…", flush=True)
    t0 = time.time()
    base = loader.load_base_model()
    print(f"Loaded {base.model_id} in {time.time()-t0:.1f}s\n")
    print(f"Prompt: {PROMPT_BASE}\n")
    t0 = time.time()
    reply = stream_and_print(base, [{"role": "user", "content": PROMPT_BASE}], 150, True)
    print(f"  [{time.time()-t0:.1f}s]")
    result(bool(reply.strip()))

    # ── Test 3: NLI Core (fine-tuned 1.7B, thinking OFF) ──────────────────────
    banner("Test 3 — NLI Core  (fine-tuned 1.7B · thinking=OFF)")
    print("Loading…", flush=True)
    t0 = time.time()
    nli = loader.load_nli_model()
    print(f"Loaded {nli.model_id} in {time.time()-t0:.1f}s\n")
    print(f"Prompt: {PROMPT_NLI[:100]}…\n")
    t0 = time.time()
    reply = stream_and_print(
        nli,
        [
            {"role": "system", "content": "You are a contract NLI system. Return ONLY a JSON array."},
            {"role": "user",   "content": PROMPT_NLI},
        ],
        150,
        False,
    )
    print(f"  [{time.time()-t0:.1f}s]")
    passed = "H04" in reply and any(
        lbl in reply for lbl in ("ENTAILED", "CONTRADICTED", "NOT_MENTIONED")
    )
    result(passed)

    print(f"\n{'='*55}\n  {BOLD}All tests complete.{RESET}\n{'='*55}\n")
