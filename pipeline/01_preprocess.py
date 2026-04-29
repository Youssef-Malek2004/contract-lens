"""
Build SFT dataset from ContractNLI train.json.

Usage:
    python scripts/01_preprocess.py

Reads:  data/train.json
Writes: data/train.jsonl
        data/valid.jsonl
"""
import sys
from pathlib import Path

# Allow `src` imports when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import make_dataset

DATA_DIR = Path(__file__).parent.parent / "data"

if __name__ == "__main__":
    print("Building SFT dataset...")
    make_dataset(
        train_json_path=str(DATA_DIR / "train.json"),
        output_dir=str(DATA_DIR),
        val_split=0.1,
        seed=42,
        test_json_path=str(DATA_DIR / "test.json"),
    )
    print("\nDone.")
