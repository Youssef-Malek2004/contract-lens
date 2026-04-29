#!/usr/bin/env python3
"""
quick_infer.py
Run the fine-tuned Qwen3 model on one validation contract on Mac (MPS/CPU).
No unsloth required — uses plain transformers.

Usage:
    python quick_infer.py                   # first val doc
    python quick_infer.py --idx 3           # 4th val doc
    python quick_infer.py --data data/train.json --idx 0
"""

import argparse
import itertools
import json
import random
import re
import sys
import threading
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID     = "Youssef-Malek/contractnli-vast-ai-qwen3-1.7b"
DATA_PATH    = "data/train.json"
VAL_SPLIT    = 0.1
SEED         = 42
MAX_NEW_TOK  = 2048
CONF_NEW_TOK = 512

# ── Inline constants (mirrors src/constants.py) ──────────────────────────────
NDA_TO_H = {
    "nda-1":  "H01", "nda-2":  "H02", "nda-3":  "H03", "nda-4":  "H04",
    "nda-5":  "H05", "nda-7":  "H06", "nda-8":  "H07", "nda-10": "H08",
    "nda-11": "H09", "nda-12": "H10", "nda-13": "H11", "nda-15": "H12",
    "nda-16": "H13", "nda-17": "H14", "nda-18": "H15", "nda-19": "H16",
    "nda-20": "H17",
}

LABEL_MAP = {
    "Entailment":   "ENTAILED",
    "Contradiction": "CONTRADICTED",
    "NotMentioned": "NOT_MENTIONED",
}

HYPOTHESES = {
    "H01": "All Confidential Information shall be expressly identified by the Disclosing Party.",
    "H02": "Confidential Information shall only include technical information.",
    "H03": "Confidential Information may include verbally conveyed information.",
    "H04": "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.",
    "H05": "Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
    "H06": "Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).",
    "H07": "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.",
    "H08": "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.",
    "H09": "Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.",
    "H10": "Receiving Party may independently develop information similar to Confidential Information.",
    "H11": "Receiving Party may acquire information similar to Confidential Information from a third party.",
    "H12": "Agreement shall not grant Receiving Party any right to Confidential Information.",
    "H13": "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.",
    "H14": "Receiving Party may create a copy of some Confidential Information in some circumstances.",
    "H15": "Receiving Party shall not solicit some of Disclosing Party's representatives.",
    "H16": "Some obligations of Agreement may survive termination of Agreement.",
    "H17": "Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.",
}

SYSTEM_PROMPT = (
    "You are a contract NLI system. Given a list of numbered contract spans and "
    "a set of hypotheses, classify each hypothesis as one of:\n"
    "- ENTAILED: the contract entails the hypothesis\n"
    "- CONTRADICTED: the contract contradicts the hypothesis\n"
    "- NOT_MENTIONED: the contract does not mention this\n\n"
    "For ENTAILED and CONTRADICTED, you must identify the span IDs that serve as "
    "evidence. Pay careful attention to exceptions introduced by phrases like "
    '"notwithstanding the foregoing", "except", "provided however" — these can '
    "flip the meaning of an earlier general rule.\n\n"
    "Return ONLY a JSON array. No other text, no markdown fences."
)

CONF_SYSTEM = (
    "You are a contract NLI auditor. Given the NLI results for a contract, "
    "provide a confidence score (0.0-1.0) for each hypothesis decision. "
    "For ENTAILED or CONTRADICTED decisions, also provide a verbatim quote "
    "from the contract that supports the decision.\n\n"
    'Return ONLY a JSON object keyed by hypothesis ID, e.g. '
    '{"H01": {"confidence": 0.95, "quote": "..."}, ...}. '
    "No other text."
)

# ── Data helpers ─────────────────────────────────────────────────────────────

def load_val_doc(data_path: str, idx: int) -> dict:
    with open(data_path) as f:
        data = json.load(f)
    docs = data["documents"]
    rng = random.Random(SEED)
    rng.shuffle(docs)
    n_val = int(len(docs) * VAL_SPLIT)
    val_docs = docs[:n_val]
    if idx >= len(val_docs):
        sys.exit(f"idx {idx} out of range — only {len(val_docs)} val docs available.")
    return val_docs[idx]


def build_chunks(doc: dict) -> list[dict]:
    text = doc["text"]
    chunks = []
    for i, (start, end) in enumerate(doc["spans"]):
        span_text = text[start:end].strip()
        if span_text:
            chunks.append({"original_index": i, "text": span_text})
    return chunks


def build_user_prompt(doc: dict) -> str:
    chunks = build_chunks(doc)
    span_lines = "\n".join(f'[{c["original_index"]}] "{c["text"]}"' for c in chunks)
    hyp_lines  = "\n".join(f"{h}: {HYPOTHESES[h]}" for h in sorted(HYPOTHESES))
    return f"Contract Spans:\n{span_lines}\n\nHypotheses:\n{hyp_lines}"


def gold_labels(doc: dict) -> dict[str, str]:
    annotations = doc["annotation_sets"][0]["annotations"]
    result = {}
    for nda_key, h_id in NDA_TO_H.items():
        ann = annotations.get(nda_key, {"choice": "NotMentioned"})
        result[h_id] = LABEL_MAP[ann["choice"]]
    return result

# ── Inference helpers ─────────────────────────────────────────────────────────

class _Spinner:
    """Animates a braille spinner in a background thread."""
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str):
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        for ch in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            print(f"\r{ch} {self.message}", end="", flush=True)
            time.sleep(0.08)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        # clear the spinner line
        print(f"\r{' ' * (len(self.message) + 4)}\r", end="", flush=True)


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int, device: str, label: str = "") -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300.0
    )

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    # generation runs in a background thread; main thread drives the UI
    gen_thread = threading.Thread(target=lambda: model.generate(**gen_kwargs))
    gen_thread.start()

    # spinner while the model is doing prefill (before first token arrives)
    header = label or "Generating"
    with _Spinner(f"Reading {input_len} input tokens…"):
        first_chunk = next(iter(streamer))  # blocks until prefill is done

    # stream remaining tokens to stdout
    print(f"{header} ▶ ", end="", flush=True)
    print(first_chunk, end="", flush=True)
    output = first_chunk
    for chunk in streamer:
        print(chunk, end="", flush=True)
        output += chunk

    print()  # newline after last token
    gen_thread.join()
    return output.strip()


def parse_nli(raw: str):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # try to find a JSON array anywhere in the output
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def parse_confidence(raw: str):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        # model sometimes returns an array with hypothesis_id + confidence fields
        if isinstance(data, list):
            result = {}
            for entry in data:
                h_id = entry.get("hypothesis_id")
                if h_id:
                    result[h_id] = {
                        "confidence": entry.get("confidence"),
                        "quote":      entry.get("quote", ""),
                    }
            return result or None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx",  type=int, default=0, help="Validation doc index (0-based)")
    parser.add_argument("--data", type=str, default=DATA_PATH)
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    # Adapter repo only has LoRA weights trained on unsloth/Qwen3-1.7B-bnb-4bit
    # (CUDA-only). Load the equivalent non-quantized base for Mac, then apply
    # the adapter on top via PEFT.
    BASE_MODEL = "Qwen/Qwen3-1.7B"
    print(f"Loading base model: {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)  # adapter repo has the patched chat template
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print(f"Applying LoRA adapter: {MODEL_ID} ...")
    model = PeftModel.from_pretrained(base, MODEL_ID)
    model = model.to(device)
    model.eval()
    print("Model loaded.")
    print("  Agent 1 → fine-tuned  (adapters ON  — NLI classification)")
    print("  Agent 2 → base model  (adapters OFF — confidence + quotes)")
    print("  Switching between agents is instant (same weights, adapter toggled in-place)\n")

    # ── Load doc ──────────────────────────────────────────────────────────────
    doc = load_val_doc(args.data, args.idx)
    contract_id = doc.get("id", f"doc_{args.idx}")
    print(f"Contract: {contract_id}  (val idx={args.idx})")
    print(f"Spans: {len(doc['spans'])}  |  Text length: {len(doc['text'])} chars\n")
    gold = gold_labels(doc)

    # ── Pass 1: NLI ───────────────────────────────────────────────────────────
    user_prompt = build_user_prompt(doc)
    messages_p1 = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_prompt},
    ]

    print("=" * 60)
    print("AGENT 1 [fine-tuned] — NLI classification")
    print("=" * 60)
    t0 = time.time()
    raw_nli = generate(model, tokenizer, messages_p1, MAX_NEW_TOK, device, label="Agent 1 [fine-tuned]")
    t1 = time.time()
    print(f"\nRaw output ({t1-t0:.1f}s):\n{raw_nli}\n")

    nli_results = parse_nli(raw_nli)
    if nli_results is None:
        print("WARNING: could not parse NLI JSON — aborting pass 2.")
        sys.exit(1)

    # ── Pass 2: base model (adapters disabled — instant toggle) ───────────────
    nli_summary = json.dumps(nli_results, indent=2)
    conf_user = (
        f"Here are the NLI results for the contract:\n{nli_summary}\n\n"
        "For each hypothesis, provide your confidence score (0.0-1.0) and, "
        "for ENTAILED or CONTRADICTED, a verbatim quote from the contract."
    )
    messages_p2 = [
        {"role": "system",    "content": CONF_SYSTEM},
        {"role": "user",      "content": conf_user},
    ]

    print("=" * 60)
    print("AGENT 2 [base model] — Confidence + quotes  (adapters disabled)")
    print("=" * 60)
    t2 = time.time()
    with model.disable_adapter():
        raw_conf = generate(model, tokenizer, messages_p2, CONF_NEW_TOK, device, label="Agent 2 [base]")
    t3 = time.time()
    print(f"\nRaw output ({t3-t2:.1f}s):\n{raw_conf}\n")

    conf_results = parse_confidence(raw_conf)

    # ── Results table ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("RESULTS vs GOLD")
    print("=" * 60)
    print(f"{'HYP':<5} {'GOLD':<15} {'PRED':<15} {'CONF':<6}  MATCH")
    print("-" * 60)

    correct = 0
    for h_id in sorted(HYPOTHESES):
        gold_label = gold.get(h_id, "?")

        pred_entry = next((e for e in nli_results if e.get("hypothesis_id") == h_id), None)
        pred_label = pred_entry.get("label", "?") if pred_entry else "MISSING"

        conf_entry = conf_results.get(h_id, {}) if conf_results else {}
        conf_val   = conf_entry.get("confidence", "-")
        conf_str   = f"{conf_val:.2f}" if isinstance(conf_val, float) else str(conf_val)

        match = "OK" if pred_label == gold_label else "WRONG"
        if match == "OK":
            correct += 1
        print(f"{h_id:<5} {gold_label:<15} {pred_label:<15} {conf_str:<6}  {match}")

    total = len(HYPOTHESES)
    print("-" * 60)
    print(f"Accuracy: {correct}/{total} = {correct/total:.1%}")
    print(f"Total wall time: {t3-t0:.1f}s")


if __name__ == "__main__":
    main()
