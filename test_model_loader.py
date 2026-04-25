#!/usr/bin/env python3
"""
test_model_loader.py
Smoke-test all models through model_loader.py with live token streaming.
Thinking tokens stream in dim colour; response tokens stream normally.

Usage:
    conda activate genai-ms2
    python test_model_loader.py
"""
import threading
import time

import torch
from transformers import TextIteratorStreamer

from src.model_loader import get_device, load_orchestrator, load_nli_model

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

# ANSI colour helpers
DIM   = "\033[2m"
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[32m"
RED   = "\033[31m"


def generate_streamed(model, tokenizer, messages, max_new_tokens=150, thinking=True) -> str:
    """
    Run generation in a background thread, stream tokens live.
    Thinking tokens (<think>…</think>) print dimmed.
    Returns the full raw output string.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    print(f"  {DIM}[{input_len} input tokens]{RESET}  ", end="", flush=True)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,   # keep <think> tags so we can detect them
        timeout=300.0,
    )

    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        ),
    )
    thread.start()

    full = ""
    in_think = False
    for chunk in streamer:
        full += chunk
        # toggle dim rendering inside <think>…</think>
        if "<think>" in chunk:
            in_think = True
            print(DIM, end="", flush=True)
        if "</think>" in chunk:
            in_think = False
            print(chunk + RESET, end="", flush=True)
            continue
        print(chunk, end="", flush=True)

    if in_think:          # model stopped mid-think
        print(RESET, end="", flush=True)
    print()               # newline after generation
    thread.join()

    # strip special tokens for return value
    clean = tokenizer.decode(
        tokenizer.encode(full, add_special_tokens=False),
        skip_special_tokens=True,
    ).strip()
    return clean


def banner(title):
    print(f"\n{'='*55}\n  {BOLD}{title}{RESET}\n{'='*55}")


def result(passed: bool):
    if passed:
        print(f"{GREEN}  PASS{RESET}")
    else:
        print(f"{RED}  FAIL{RESET}")


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {BOLD}{device}{RESET}")

    # ── Test 1: Orchestrator (Qwen3.5-4B, thinking ON) ───────────────────────
    banner("Test 1 — Orchestrator  (Qwen3-4B · thinking=ON)")
    print("Loading...", flush=True)
    t0 = time.time()
    orch_model, orch_tok = load_orchestrator(device)
    print(f"Loaded in {time.time()-t0:.1f}s\n")

    print(f"Prompt: {PROMPT_ORCHESTRATOR}\n")
    t0 = time.time()
    reply = generate_streamed(
        orch_model, orch_tok,
        [{"role": "user", "content": PROMPT_ORCHESTRATOR}],
        max_new_tokens=200, thinking=True,
    )
    print(f"\n  [{time.time()-t0:.1f}s]")
    result(bool(reply))

    # ── Test 2: Base model (adapter OFF, thinking ON) ─────────────────────────
    banner("Test 2 — Base model  (Qwen3-1.7B · adapter OFF · thinking=ON)")
    print("Loading NLI model (base + adapter)...", flush=True)
    t0 = time.time()
    nli_model, nli_tok = load_nli_model(device)
    print(f"Loaded in {time.time()-t0:.1f}s\n")

    print(f"Prompt: {PROMPT_BASE}\n")
    t0 = time.time()
    with nli_model.disable_adapter():
        reply = generate_streamed(
            nli_model, nli_tok,
            [{"role": "user", "content": PROMPT_BASE}],
            max_new_tokens=150, thinking=True,
        )
    print(f"\n  [{time.time()-t0:.1f}s]")
    result(bool(reply))

    # ── Test 3: NLI Core (adapter ON, thinking OFF) ───────────────────────────
    banner("Test 3 — NLI Core  (Qwen3-1.7B · adapter ON · thinking=OFF)")
    print(f"Prompt: {PROMPT_NLI[:100]}...\n")
    t0 = time.time()
    reply = generate_streamed(
        nli_model, nli_tok,
        [{"role": "system", "content": "You are a contract NLI system. Return ONLY a JSON array."},
         {"role": "user",   "content": PROMPT_NLI}],
        max_new_tokens=150, thinking=False,
    )
    print(f"\n  [{time.time()-t0:.1f}s]")
    passed = "H04" in reply and any(
        lbl in reply for lbl in ("ENTAILED", "CONTRADICTED", "NOT_MENTIONED")
    )
    result(passed)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {BOLD}All tests complete.{RESET}")
    print(f"{'='*55}\n")
