"""
05b_debug_single.py
====================
Run the two-pass pipeline on a single test contract and print every raw model
output in full — including <think> blocks — so you can see what the model is
actually doing before parsing kicks in.

Usage:
    python scripts/05b_debug_single.py              # uses contract index 0
    python scripts/05b_debug_single.py --idx 42     # any index 0-122
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from unsloth import FastLanguageModel

# ── Config ────────────────────────────────────────────────────────────────
MODEL         = "Youssef-Malek/contractnli-vast-ai-qwen3-1.7b"
TEST_JSONL    = "./test.jsonl"
TEST_JSON     = "./test.json"
MAX_SEQ       = 13288
MAX_NEW_TOK   = 2048
CONF_NEW_TOK  = 1024
# ─────────────────────────────────────────────────────────────────────────

CONFIDENCE_QUESTION = (
    "For each of your 17 classifications above provide two things:\n"
    "1. confidence: your confidence score from 0.0 to 1.0.\n"
    "2. quote: for ENTAILED or CONTRADICTED, copy the single most relevant verbatim "
    "excerpt from the contract text you used as evidence. For NOT_MENTIONED leave quote empty.\n\n"
    "Reply ONLY with a JSON object, all 17 hypotheses:\n"
    '{"H01": {"confidence": 0.95, "quote": "verbatim excerpt..."}, '
    '"H02": {"confidence": 0.70, "quote": ""}, ...}'
)

SEP = "─" * 70


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def split_prompt_and_gold(text):
    for tag in (
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|im_start|>assistant\n",
    ):
        idx = text.find(tag)
        if idx != -1:
            return text[: idx + len(tag)], text[idx + len(tag):].replace("<|im_end|>", "").strip()
    return text, ""


def generate(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ - max_new_tokens,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Decode WITHOUT stripping special tokens so <think> tags are visible
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    # Strip padding/eos tokens but keep <think> content
    raw = raw.replace(tokenizer.eos_token or "", "").replace("<|im_end|>", "").strip()
    return raw


def print_section(title, content):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    print(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Test contract index (0-122)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    rows = load_jsonl(TEST_JSONL)
    with open(TEST_JSON) as f:
        docs = json.load(f)["documents"]

    if args.idx >= len(rows):
        print(f"Index {args.idx} out of range (0-{len(rows)-1})")
        sys.exit(1)

    row = rows[args.idx]
    doc = docs[args.idx]
    prompt, gold_raw = split_prompt_and_gold(row["text"])

    print(f"\n{'='*70}")
    print(f"  CONTRACT INDEX : {args.idx}")
    print(f"  CONTRACT ID    : {doc.get('id', 'unknown')}")
    print(f"  SPANS          : {len(doc['spans'])}")
    print(f"  TEXT LENGTH    : {len(doc['text'])} chars")
    print(f"{'='*70}")

    print_section("GOLD ANSWER (ground truth)", gold_raw)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        fast_inference=False,
        attn_implementation="flash_attention_2",
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.\n")

    # ── Pass 1: NLI inference ─────────────────────────────────────────────
    print(f"{SEP}\n  PASS 1 — NLI INFERENCE\n{SEP}")
    print("Running...")
    pass1_raw = generate(model, tokenizer, prompt, MAX_NEW_TOK)

    print_section("PASS 1 RAW OUTPUT (full, including <think>)", pass1_raw)

    # Split think from answer for readability
    think_match = re.search(r"<think>(.*?)</think>(.*)", pass1_raw, re.DOTALL)
    if think_match:
        print_section("PASS 1 — THINK BLOCK", think_match.group(1).strip())
        print_section("PASS 1 — ANSWER (after think)", think_match.group(2).strip())
    else:
        print("\n  (no <think> block detected in pass 1)")

    # ── Pass 2: Confidence + quote elicitation ────────────────────────────
    conf_prompt = (
        prompt
        + pass1_raw
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + CONFIDENCE_QUESTION
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + "<think>\nLet me assess the evidence quality for each hypothesis:\n"
    )

    print(f"\n{SEP}\n  PASS 2 — CONFIDENCE + QUOTE ELICITATION\n{SEP}")
    print("Running...")
    pass2_raw = generate(model, tokenizer, conf_prompt, CONF_NEW_TOK)

    print_section("PASS 2 RAW OUTPUT (full, including <think>)", pass2_raw)

    # Seeded think: decoded output starts inside the think block (no opening tag),
    # model closes it with </think> then outputs the JSON answer.
    seeded_match = re.search(r"^(.*?)</think>(.*)", pass2_raw, re.DOTALL)
    full_match   = re.search(r"<think>(.*?)</think>(.*)", pass2_raw, re.DOTALL)
    think_match2 = full_match or seeded_match

    if think_match2:
        label = "PASS 2 — THINK BLOCK (seeded)" if not full_match else "PASS 2 — THINK BLOCK"
        print_section(label, think_match2.group(1).strip())
        print_section("PASS 2 — ANSWER (after think)", think_match2.group(2).strip())
        answer_text = think_match2.group(2).strip()
    else:
        print("\n  (no <think> block detected in pass 2)")
        answer_text = pass2_raw

    # ── Quick parse check ─────────────────────────────────────────────────
    try:
        cleaned = re.sub(r"```json|```", "", answer_text).strip()
        parsed = json.loads(cleaned)
        # Handle array format [{hypothesis_id, confidence, quote}] the model defaults to
        if isinstance(parsed, list):
            parsed = {
                item["hypothesis_id"]: item
                for item in parsed
                if isinstance(item, dict) and re.match(r"^H(0[1-9]|1[0-7])$", str(item.get("hypothesis_id", "")))
            }
        print(f"\n{SEP}\n  PARSED CONFIDENCE + QUOTES\n{SEP}")
        for hyp_id in sorted(parsed.keys()):
            entry = parsed[hyp_id]
            conf  = entry.get("confidence", "?")
            quote = entry.get("quote", "")
            quote_ok = (quote in doc["text"]) if quote else None
            integrity = "✓ verbatim match" if quote_ok is True else ("✗ NOT found in contract" if quote_ok is False else "— no quote")
            print(f"  {hyp_id}  conf={conf:<6}  {integrity}")
            if quote:
                print(f"         quote: \"{quote[:80]}{'...' if len(quote) > 80 else ''}\"")
    except Exception as e:
        print(f"\n  [!] JSON parse failed on pass 2 answer: {e}")
        print(f"      Raw answer text: {answer_text[:500]}")

    print(f"\n{'='*70}\n  Done.\n{'='*70}\n")


if __name__ == "__main__":
    main()
