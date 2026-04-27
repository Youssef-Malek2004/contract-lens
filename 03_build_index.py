#!/usr/bin/env python3
"""
03_build_index.py — Shared index builder for Vector RAG and GraphRAG.

Usage:
    python 03_build_index.py --mode vector          # build FAISS index only
    python 03_build_index.py --mode graph           # build graph index only
    python 03_build_index.py --mode all             # build both

Options:
    --train-path PATH   Path to training data (default: data/train.json)
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build retrieval indexes for ContractLens MS2"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["vector", "graph", "all"],
        help="Which index to build: vector | graph | all",
    )
    parser.add_argument(
        "--train-path",
        default="data/train.json",
        help="Path to training JSON (default: data/train.json)",
    )
    args = parser.parse_args()

    start = time.time()

    if args.mode in ("vector", "all"):
        print("=" * 60)
        print("  Building Vector RAG index")
        print("=" * 60)
        from src.rag_vector import build_index as build_vector
        build_vector(train_path=args.train_path)

    if args.mode in ("graph", "all"):
        print("=" * 60)
        print("  Building GraphRAG index")
        print("=" * 60)
        from src.rag_graph import build_index as build_graph
        build_graph(train_path=args.train_path)

    elapsed = time.time() - start
    print(f"\nAll done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
