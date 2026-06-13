#!/usr/bin/env python3
"""
download_models.py
Pre-download all models into the HuggingFace cache (file download only —
does NOT load anything into memory, avoids quantization config issues).

RUN AS:
    conda activate genai-ms2
    python download_models.py          # NOT via conda run (buffers progress bars)
"""
from huggingface_hub import snapshot_download

MODELS = [
    {
        "repo_id": "Qwen/Qwen3-1.7B",
        "desc":    "NLI Core base + Hypothesis Agent weights",
        "size":    "~3.5 GB",
    },
    {
        "repo_id": "Qwen/Qwen3-4B",
        "desc":    "Orchestrator + Conversation Agent",
        "size":    "~8.0 GB",
    },
    {
        "repo_id": "Youssef-Malek/contractnli-vast-ai-qwen3-1.7b",
        "desc":    "Fine-tuned LoRA adapter (NLI Core)",
        "size":    "~100 MB",
        "ignore":  ["optimizer.pt", "rng_state.pth", "scheduler.pt",
                    "training_args.bin", "checkpoint-*/optimizer.pt",
                    "checkpoint-*/rng_state.pth", "checkpoint-*/scheduler.pt",
                    "checkpoint-*/training_args.bin"],
    },
    {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "desc":    "Vector RAG embeddings",
        "size":    "~90 MB",
    },
]

if __name__ == "__main__":
    print("ContractLens MS2 — Model Download")
    print("Downloads files only — no loading into memory.\n")

    for m in MODELS:
        print(f"{'='*60}")
        print(f"  {m['repo_id']}  ({m['size']})")
        print(f"  {m['desc']}")
        print(f"{'='*60}")

        kwargs = {"repo_id": m["repo_id"]}
        if "ignore" in m:
            kwargs["ignore_patterns"] = m["ignore"]

        snapshot_download(**kwargs)
        print(f"  Done.\n")

    print("All models cached.")
