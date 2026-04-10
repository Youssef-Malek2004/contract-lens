#!/bin/bash
# Fine-tune Qwen3.5-9B-4bit on ContractNLI SFT data using LoRA.
# Run from repo root: bash scripts/02_finetune.sh

python -m mlx_lm.lora \
    --model mlx-community/Qwen3.5-9B-4bit \
    --data data \
    --train \
    --batch-size 1 \
    --num-layers 16 \
    --iters 1269 \
    --learning-rate 2e-4 \
    --steps-per-eval 100 \
    --val-batches 20 \
    --max-seq-length 16384 \
    --grad-checkpoint \
    --adapter-path ./adapters
