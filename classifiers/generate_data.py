#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifiers/generate_data.py
Unified training data generator for:
  - animacy (animate / inanimate)
  - definiteness (definite / indefinite)
  - pronominality (pronoun / common)

Usage
------
# Default settings (balanced sampling, max=4000, split=valid)
python -m classifiers.generate_data --task animacy

# Random sampling example on the train split
python -m classifiers.generate_data --task pronominality --max 200 --strategy random
"""

import os
import json
import argparse
import csv
import random
import time
from collections import Counter

from tqdm import tqdm
from openai import OpenAI

from utils import DATA_PATH, NP_CLASSIFICATION_PROMPTS

client = OpenAI()

def parse_args():
    ap = argparse.ArgumentParser(description="Generate training data for semantic NP classifiers.")
    ap.add_argument("--task", choices=["animacy", "definiteness", "pronominality"], required=True,
                    help="Semantic classification task.")
    ap.add_argument("--max", type=int, default=4000,
                    help="Maximum number of labeled examples to generate.")
    ap.add_argument("--strategy", choices=["random", "balanced"], default="balanced",
                    help="Sampling strategy: 'balanced' enforces per-class limits.")
    ap.add_argument("--split", choices=["train", "valid", "test"], default="valid",
                    help="Which data split to use (matched by *_<split>_verbs.jsonl).")
    ap.add_argument("--model", type=str, default="gpt-4o",
                    help="OpenAI model name.")
    return ap.parse_args()

TASK_CONFIGS = {
    "animacy": {
        "field": "animacy",
        "classes": ["animate", "inanimate"],
    },
    "definiteness": {
        "field": "definiteness",
        "classes": ["definite", "indef"],
    },
    "pronominality": {
        "field": "pronominality",
        "classes": ["pronoun", "common"],
    },
}

def call_openai_chat(sentence: str, np: str, np_type: str, task: str, model: str) -> str | None:
    """Call OpenAI chat completion to classify a noun phrase."""
    prompt = NP_CLASSIFICATION_PROMPTS[task].format(sentence=sentence, np_type=np_type, np=np)
    allowed = set(TASK_CONFIGS[task]["classes"])

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2,
            )
            result = (resp.choices[0].message.content or "").strip().lower()
            result = result.replace('"', "").split()[0]

            if result in allowed:
                return result
            return None
        except Exception as e:
            print(f"[WARN] OpenAI call failed ({attempt + 1}/3): {e}")
            time.sleep(3)

    return None

def ensure_csv(path: str, field: str):
    """Ensure output CSV exists and has a header row."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=["sentence", "np", "np_role", field])
            writer.writeheader()

def find_input_file(structured_dir: str, split: str) -> str:
    """
    Find a single *_<split>_verbs.jsonl file under data/structured, e.g.:
      1_train_verbs.jsonl, 1_valid_verbs.jsonl, 1_test_verbs.jsonl
    """
    suffix = f"_{split}_verbs.jsonl"
    candidates = [f for f in os.listdir(structured_dir) if f.endswith(suffix)]
    if not candidates:
        raise FileNotFoundError(f"No *{suffix} under {structured_dir}")
    if len(candidates) > 1:
        raise RuntimeError(f"Found multiple candidates for *{suffix}: {candidates}")
    return os.path.join(structured_dir, candidates[0])


def extract_training_data(task: str, max_instances: int, strategy: str, split: str, model: str):
    cfg = TASK_CONFIGS[task]

    structured_dir = os.path.join(DATA_PATH, "structured")
    input_file = find_input_file(structured_dir, split)

    stem = os.path.splitext(os.path.basename(input_file))[0]
    out_csv = os.path.join(DATA_PATH, f"training_data_{task}_{stem.replace('_verbs', '')}.csv")
    ensure_csv(out_csv, cfg["field"])

    # Step 1: collect candidate NPs
    candidates = []
    with open(input_file, encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)

            tokens = data.get("tokens", [])
            verbs = data.get("verbs") or []
            if not tokens or not verbs:
                continue

            sentence = " ".join(tok.get("text", "") for tok in tokens)

            for entry in verbs:
                subj = entry.get("subject")
                objs = entry.get("objects") or []
                if not subj or len(objs) != 1:
                    continue
                if objs[0].get("dep") == "ccomp":
                    continue

                candidates.append((sentence, subj["text"], "subject"))
                candidates.append((sentence, objs[0]["text"], "object"))

    print(f"[INFO] Using file: {input_file}")
    print(f"[INFO] Collected {len(candidates):,} candidate NPs")
    random.shuffle(candidates)

    # Step 2: sample and call model
    cls_counter = Counter()

    with open(out_csv, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["sentence", "np", "np_role", cfg["field"]])
        pbar = tqdm(total=max_instances, desc=f"Generating {task} data")

        i = 0
        while sum(cls_counter.values()) < max_instances and i < len(candidates):
            sent, np_text, role = candidates[i]
            i += 1

            label = call_openai_chat(sent, np_text, role.capitalize(), task, model)
            if not label:
                continue

            if strategy == "balanced":
                per_class_limit = max_instances // len(cfg["classes"])
                if cls_counter[label] >= per_class_limit:
                    continue

            writer.writerow(
                {
                    "sentence": sent,
                    "np": np_text,
                    "np_role": role,
                    cfg["field"]: label,
                }
            )
            fout.flush()

            cls_counter[label] += 1
            pbar.update(1)
            pbar.set_postfix({c: cls_counter.get(c, 0) for c in cfg["classes"]})

            if i > len(candidates) * 2:
                print("[WARN] Sampling stopped: exceeded 2× candidate size.")
                break

    print(f"[DONE] Wrote {sum(cls_counter.values())} examples to {out_csv}")
    print("[STATS]", dict(cls_counter))

    return out_csv

def main():
    args = parse_args()
    extract_training_data(
        task=args.task,
        max_instances=args.max,
        strategy=args.strategy,
        split=args.split,
        model=args.model,
    )

if __name__ == "__main__":
    main()
