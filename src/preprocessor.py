"""
Converts ContractNLI documents into Qwen3 chat-formatted SFT examples.
"""
import json
import random
from pathlib import Path

from src.constants import NDA_TO_H, LABEL_MAP, HYPOTHESES, SYSTEM_PROMPT


def build_chunks(doc: dict) -> list[dict]:
    """
    Extract non-empty spans from a document.
    Preserves the original span index as chunk_id so evidence_spans from
    model output map correctly even when empty spans are skipped.
    """
    text = doc["text"]
    chunks = []
    for i, (start, end) in enumerate(doc["spans"]):
        span_text = text[start:end].strip()
        if span_text:
            chunks.append({
                "chunk_id": f"chunk_{i:03d}",
                "original_index": i,
                "text": span_text,
                "span": {"char_start": start, "char_end": end},
            })
    return chunks


def build_prompt(doc: dict) -> str:
    """
    Build the user message: numbered spans followed by all 17 hypotheses.
    Uses original span indices as labels (not sequential chunk numbers).
    """
    chunks = build_chunks(doc)
    span_lines = "\n".join(
        f'[{c["original_index"]}] "{c["text"]}"' for c in chunks
    )
    hyp_lines = "\n".join(
        f"{h_id}: {HYPOTHESES[h_id]}" for h_id in sorted(HYPOTHESES)
    )
    return f"Contract Spans:\n{span_lines}\n\nHypotheses:\n{hyp_lines}"


def build_answer(doc: dict) -> str:
    """
    Build the assistant JSON answer from ground-truth annotations,
    sorted H01→H17 with mapped labels.
    """
    annotations = doc["annotation_sets"][0]["annotations"]
    answer = []
    for nda_key, h_id in sorted(NDA_TO_H.items(), key=lambda x: x[1]):
        ann = annotations.get(nda_key, {"choice": "NotMentioned", "spans": []})
        answer.append({
            "hypothesis_id": h_id,
            "label": LABEL_MAP[ann["choice"]],
            "evidence_spans": ann["spans"],
        })
    return json.dumps(answer, indent=2)


def format_training_example(doc: dict, tokenizer) -> str:
    """
    Assemble a full Qwen3 chat-template string for one document.
    Returns the raw string (not tokenized) for writing to .jsonl.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(doc)},
        {"role": "assistant", "content": build_answer(doc)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def make_dataset(
    train_json_path: str,
    output_dir: str,
    val_split: float = 0.1,
    seed: int = 42,
    test_json_path: str | None = None,
) -> None:
    """
    Read train.json, split into train/val, write .jsonl files.
    Optionally also write test.jsonl from a separate test_json_path.
    Each line: {"text": "<chat-formatted string>"}
    """
    from mlx_lm import load as mlx_load

    print("Loading tokenizer...")
    _, tokenizer = mlx_load("mlx-community/Qwen3.5-9B-4bit")

    with open(train_json_path) as f:
        data = json.load(f)

    docs = data["documents"]
    rng = random.Random(seed)
    rng.shuffle(docs)

    n_val = int(len(docs) * val_split)
    val_docs   = docs[:n_val]
    train_docs = docs[n_val:]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    token_lengths = []

    for split_name, split_docs in [("train", train_docs), ("valid", val_docs)]:
        path = out / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for doc in split_docs:
                text = format_training_example(doc, tokenizer)
                f.write(json.dumps({"text": text}) + "\n")
                token_lengths.append(len(tokenizer.encode(text)))
        print(f"  {split_name}: {len(split_docs)} examples → {path}")

    if test_json_path:
        with open(test_json_path) as f:
            test_data = json.load(f)
        test_docs = test_data["documents"]
        path = out / "test.jsonl"
        with open(path, "w") as f:
            for doc in test_docs:
                text = format_training_example(doc, tokenizer)
                f.write(json.dumps({"text": text}) + "\n")
                token_lengths.append(len(tokenizer.encode(text)))
        print(f"  test:  {len(test_docs)} examples → {path}")

    all_lengths = sorted(token_lengths)
    n = len(all_lengths)
    print(f"\nToken length stats (all {n} examples):")
    print(f"  min={all_lengths[0]}")
    print(f"  p50={all_lengths[n // 2]}")
    print(f"  p90={all_lengths[int(n * 0.9)]}")
    print(f"  max={all_lengths[-1]}")
