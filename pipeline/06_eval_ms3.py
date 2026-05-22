#!/usr/bin/env python3
"""Numbered entry point for Member 5's MS3 evaluation runner."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.eval_ms3 import main


if __name__ == "__main__":
    main()
