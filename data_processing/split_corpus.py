#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_processing/split_corpus.py
Split a single corpus .txt file under $DATA_PATH/raw into train/valid/test.

Usage
------
# Default ratios: 0.9 / 0.05 / 0.05
python -m data_processing.split_corpus

# Custom ratios
python -m data_processing.split_corpus --train-ratio 0.7 --valid-ratio 0.2 --test-ratio 0.1
"""

import argparse
from pathlib import Path
import random

from utils import DATA_PATH

def parse_args():
    ap = argparse.ArgumentParser(description="Split a corpus into train/valid/test splits.")
    ap.add_argument("--train-ratio", type=float, default=0.9, help="Training set proportion.")
    ap.add_argument("--valid-ratio", type=float, default=0.05, help="Validation set proportion.")
    ap.add_argument("--test-ratio", type=float, default=0.05, help="Test set proportion.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    return ap.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(DATA_PATH) / "raw"

    # Expect exactly one corpus file under $DATA_PATH/raw
    txt_files = list(raw_dir.glob("*.txt"))
    if len(txt_files) != 1:
        raise RuntimeError(f"[Error] Expected exactly 1 .txt file in {raw_dir}, found {len(txt_files)}")
    corpus_file = txt_files[0]

    print(f"[Info] Using corpus: {corpus_file}")

    base = corpus_file.stem

    # Read all non-empty lines
    with corpus_file.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"[Info] Total {len(lines):,} non-empty lines")

    # Shuffle lines
    random.seed(args.seed)
    random.shuffle(lines)

    n_total = len(lines)
    n_train = int(n_total * args.train_ratio)
    n_valid = int(n_total * args.valid_ratio)
    n_test = n_total - n_train - n_valid

    train_lines = lines[:n_train]
    valid_lines = lines[n_train:n_train + n_valid]
    test_lines = lines[n_train + n_valid:]

    out_train = raw_dir / f"{base}_train.txt"
    out_valid = raw_dir / f"{base}_valid.txt"
    out_test = raw_dir / f"{base}_test.txt"

    with out_train.open("w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    with out_valid.open("w", encoding="utf-8") as f:
        f.write("\n".join(valid_lines) + "\n")
    with out_test.open("w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    print(f"[OK] Wrote {len(train_lines):,} lines to {out_train}")
    print(f"[OK] Wrote {len(valid_lines):,} lines to {out_valid}")
    print(f"[OK] Wrote {len(test_lines):,} lines to {out_test}")


if __name__ == "__main__":
    main()
