#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_processing/filter_sentences.py
Filter corpus sentences by length using regex-based sentence splitting.

Usage
------
# Default: min_len=3, max_len=30
python -m data_processing.filter_sentences

# Custom thresholds
python -m data_processing.filter_sentences --min-len 5 --max-len 20
"""

import argparse
import sys
import re
from pathlib import Path

from utils import DATA_PATH


def parse_args():
    ap = argparse.ArgumentParser(description="Filter corpus sentences by length.")
    ap.add_argument("--min-len", type=int, default=3, help="Minimum number of tokens.")
    ap.add_argument("--max-len", type=int, default=30, help="Maximum number of tokens.")
    return ap.parse_args()

def regex_sentencize(text: str):
    """Split text into sentences using a simple regex."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def main():
    args = parse_args()

    raw_dir = Path(DATA_PATH) / "raw"
    filtered_dir = Path(DATA_PATH) / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    txt_files = [p for p in raw_dir.glob("*.txt") if any(s in p.stem for s in ("train", "valid", "test"))]
    if not txt_files:
        print(f"[Error] No matching .txt files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    total_raw_sents = total_raw_tokens = 0
    total_filt_sents = total_filt_tokens = 0

    for input_path in txt_files:
        output_path = filtered_dir / input_path.name

        print(f"[Info] Processing {input_path}")

        filtered = []
        raw_sents = raw_tokens = 0
        filt_sents = filt_tokens = 0

        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                for sent in regex_sentencize(line):
                    tokens = sent.split()
                    if not tokens:
                        continue

                    raw_sents += 1
                    raw_tokens += len(tokens)

                    if args.min_len <= len(tokens) <= args.max_len:
                        filt_sents += 1
                        filt_tokens += len(tokens)
                        filtered.append(sent)

        with output_path.open("w", encoding="utf-8") as f_out:
            for sent in filtered:
                f_out.write(sent + "\n")

        total_raw_sents += raw_sents
        total_raw_tokens += raw_tokens
        total_filt_sents += filt_sents
        total_filt_tokens += filt_tokens

    print("\n[Summary]")
    print(f"  raw_sents      : {total_raw_sents}")
    print(f"  raw_tokens     : {total_raw_tokens}")
    print(f"  kept_sents     : {total_filt_sents}")
    print(f"  kept_tokens    : {total_filt_tokens}")
    print(f"  dropped_sents  : {total_raw_sents - total_filt_sents}")
    print(f"  dropped_tokens : {total_raw_tokens - total_filt_tokens}")

if __name__ == "__main__":
    main()
