#!/usr/bin/env python3
"""
merge_adapter.py
Merge the LoRA adapter into the base model and convert both 1.7B variants
to MLX 4-bit format ready for vllm-mlx serving.

Run this once before starting the vllm NLI server. The orchestrator
(Qwen3-4B) and base model (Qwen3-1.7B) are already available as pre-built
MLX models on mlx-community — only the fine-tuned NLI model needs merging.

Steps performed:
  1. Load Qwen3-1.7B base + LoRA adapter
  2. Merge adapter weights into base (merge_and_unload)
  3. Save merged model to --output-dir (default: ./merged-nli-1.7b)
  4. If --convert: convert both merged NLI and base to MLX 4-bit
  5. If --push: upload converted models to HuggingFace Hub

Usage:
    # Merge only (inspect output before converting)
    python scripts/merge_adapter.py

    # Merge + convert to MLX 4-bit locally
    python scripts/merge_adapter.py --convert

    # Merge + convert + push to Hub
    python scripts/merge_adapter.py --convert --push --hf-user your-hf-username
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def merge(output_dir: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.loaders._constants import NLI_ADAPTER_ID, NLI_BASE_ID

    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        print(f"[merge] '{out}' already exists and is non-empty — skipping merge.")
        print(f"[merge] Delete it manually if you want to re-merge.")
        return out

    print(f"[merge] Loading base: {NLI_BASE_ID}")
    base = AutoModelForCausalLM.from_pretrained(
        NLI_BASE_ID, dtype=torch.float16, low_cpu_mem_usage=True
    )

    print(f"[merge] Applying adapter: {NLI_ADAPTER_ID}")
    model = PeftModel.from_pretrained(base, NLI_ADAPTER_ID)

    print("[merge] Merging and unloading adapter…")
    merged = model.merge_and_unload()

    print(f"[merge] Saving merged model to {out}")
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))

    # Tokenizer from the adapter repo — has the patched Qwen3 chat template
    print("[merge] Saving tokenizer from adapter repo…")
    AutoTokenizer.from_pretrained(NLI_ADAPTER_ID).save_pretrained(str(out))

    # transformers-from-source omits rope_theta when serialising Qwen3Config,
    # but mlx_lm's ModelArgs requires it. Patch config.json from the in-memory
    # base config before the merged directory is used for anything else.
    _patch_rope_theta(out, base.config)

    print(f"[merge] Done → {out}")
    return out


def _patch_rope_theta(model_dir: Path, source_config) -> None:
    import json

    config_path = model_dir / "config.json"
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text())
    if "rope_theta" in config:
        return  # already present, nothing to do

    rope_theta = getattr(source_config, "rope_theta", None)
    if rope_theta is None:
        # Hard fallback: Qwen3 documented default
        rope_theta = 1_000_000.0
        print(f"[merge] Warning: rope_theta not found on base config, using default {rope_theta}")

    config["rope_theta"] = rope_theta
    config_path.write_text(json.dumps(config, indent=2))
    print(f"[merge] Patched config.json: rope_theta={rope_theta}")


def convert_to_mlx(hf_path: str, mlx_path: str, push_repo: str = None) -> None:
    import importlib.util
    import subprocess

    if importlib.util.find_spec("mlx_lm") is None:
        sys.exit(
            "[convert] mlx_lm not found in the current environment.\n"
            "  Install it with:  pip install mlx-lm\n"
            "  Then re-run:      python merge_adapter.py --convert"
        )

    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", hf_path,
        "--mlx-path", mlx_path,
        "-q", "--q-bits", "4",
    ]
    if push_repo:
        cmd += ["--upload-repo", push_repo]

    print(f"[convert] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        sys.exit(
            f"[convert] mlx_lm.convert failed for {hf_path}\n"
            "  If you see 'rope_theta' or similar missing-argument errors,\n"
            "  your mlx-lm is too old to support Qwen3. Upgrade with:\n"
            "    pip install --upgrade mlx-lm"
        )
    print(f"[convert] Done → {mlx_path}" + (f" (pushed to {push_repo})" if push_repo else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model and optionally convert to MLX 4-bit."
    )
    parser.add_argument(
        "--output-dir",
        default="./merged-nli-1.7b",
        help="Directory to save the merged model (default: ./merged-nli-1.7b)",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert merged NLI model to MLX 4-bit after merging (requires mlx-lm)",
    )
    parser.add_argument(
        "--mlx-output-dir",
        default="./mlx-nli-1.7b-4bit",
        help="Directory for the MLX-converted NLI model (default: ./mlx-nli-1.7b-4bit)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the MLX model to HuggingFace Hub after conversion (requires --hf-user)",
    )
    parser.add_argument(
        "--hf-user",
        default=None,
        help="HuggingFace username for --push (e.g. Youssef-Malek)",
    )
    args = parser.parse_args()

    if args.push and not args.hf_user:
        sys.exit("--push requires --hf-user")
    if args.push and not args.convert:
        sys.exit("--push requires --convert")

    merged_path = merge(args.output_dir)

    if args.convert:
        push_repo = None
        if args.push:
            push_repo = f"{args.hf_user}/contractnli-qwen3-1.7b-mlx-4bit"
        convert_to_mlx(str(merged_path), args.mlx_output_dir, push_repo)
        print("\n[done] NLI model ready for vllm-mlx.")
        print(f"       Serve with: MODEL={args.mlx_output_dir} PORT=8002 ./ServeLM/serve.sh")


if __name__ == "__main__":
    main()
