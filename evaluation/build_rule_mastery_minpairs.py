#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/build_rule_mastery_minpairs.py
Build minimal pairs for rule mastery evaluation (localA / localP / global).

Reads:
  $DATA_PATH/structured_labeled/*_test_verbs_labeled.ok.jsonl

Outputs:
  $EVALUATION_PATH/rule_mastery/minimal_pairs/<run_id>/test_minimal_pairs.jsonl
  $EVALUATION_PATH/rule_mastery/minimal_pairs/<run_id>/summary.json

Usage
------
python -m evaluation.build_rule_mastery_minpairs --run_id localA_animacy_natural
python -m evaluation.build_rule_mastery_minpairs --run_id global_definiteness_inverse --num_pairs 2000
"""

import os
import json
import random
import argparse
from glob import glob
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from tqdm import tqdm

from utils import (
    DATA_PATH,
    EVALUATION_PATH,
    AGENT_MARK,
    PATIENT_MARK,
    should_mark_local_A,
    should_mark_local_P,
    should_mark_global,
)

FEATURES = {"animacy", "definiteness", "pronominality"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--num_pairs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--input_dir", default=os.path.join(DATA_PATH, "structured_labeled"))
    ap.add_argument("--out_dir", default=os.path.join(EVALUATION_PATH, "rule_mastery", "minimal_pairs"))
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--max_per_sentence", type=int, default=1)
    return ap.parse_args()


def is_valid_structure(entry: dict) -> bool:
    if not entry.get("subject"):
        return False
    objs = entry.get("objects") or []
    if len(objs) != 1:
        return False
    if objs[0].get("dep") == "ccomp":
        return False
    return True


def apply_spans(tokens: List[dict], spans: List[dict]) -> str:
    if not spans:
        return " ".join(t["text"] for t in tokens)

    seen = set()
    uniq = []
    for sp in spans:
        s, e = sp.get("span", (None, None))
        if not (isinstance(s, int) and isinstance(e, int)):
            continue
        key = (s, e, sp.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(sp)

    uniq.sort(key=lambda x: x["span"][0])
    out, i, n = [], 0, len(tokens)
    for sp in uniq:
        s, e = sp["span"]
        if s < i:
            continue
        while i < s and i < n:
            out.append(tokens[i]["text"])
            i += 1
        out.append(sp.get("text", ""))
        i = e + 1
    while i < n:
        out.append(tokens[i]["text"])
        i += 1
    return " ".join(out)


def strip_marks(s: str) -> str:
    return " ".join(s.replace(AGENT_MARK, "").replace(PATIENT_MARK, "").split()).strip()


def parse_run_id(run_id: str) -> Tuple[str, Optional[str], Optional[str]]:
    parts = run_id.split("_")
    rule = parts[0]
    if rule in {"baseline", "full"}:
        return rule, None, None
    if rule not in {"localA", "localP", "global"}:
        raise ValueError(f"Unsupported run_id: {run_id}")
    if len(parts) < 3:
        raise ValueError(f"run_id must look like {rule}_<feature>_<direction>")
    feature, direction = parts[1], parts[2]
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature: {feature}")
    if direction not in {"natural", "inverse"}:
        raise ValueError(f"Unknown direction: {direction}")
    return rule, feature, direction


def should_mark(rule: str, feature: str, direction: str, A: Dict[str, str], P: Dict[str, str]) -> Tuple[bool, bool]:
    if rule == "localA":
        return should_mark_local_A(A, feature, direction), False
    if rule == "localP":
        return False, should_mark_local_P(P, feature, direction)
    if rule == "global":
        return should_mark_global(A, P, feature, direction)
    raise ValueError(rule)


def iter_inputs(input_dir: str, splits: List[str]) -> List[str]:
    paths = []
    for sp in splits:
        paths.extend(sorted(glob(os.path.join(input_dir, f"*_{sp}_verbs_labeled.ok.jsonl"))))
    if not paths:
        raise FileNotFoundError(f"No *_<split>_verbs_labeled.ok.jsonl under {input_dir} for splits={splits}")
    return paths


def main():
    args = parse_args()
    random.seed(args.seed)

    rule, feature, direction = parse_run_id(args.run_id)
    if rule in {"baseline", "full"}:
        raise ValueError("rule mastery minimal pairs are defined for localA/localP/global only.")

    inputs = iter_inputs(args.input_dir, args.splits)
    out_root = Path(args.out_dir) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    should_pool = []
    shouldnot_pool = []

    for ip in inputs:
        with open(ip, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc=os.path.basename(ip)):
                if not line.strip():
                    continue
                data = json.loads(line)
                tokens = data.get("tokens") or []
                verbs = data.get("verbs") or []
                if not tokens or not verbs:
                    continue

                picked = 0
                for entry in verbs:
                    if picked >= args.max_per_sentence:
                        break
                    if not is_valid_structure(entry):
                        continue

                    subj = entry["subject"]
                    obj = entry["objects"][0]

                    A = {
                        "animacy": subj.get("animacy"),
                        "definiteness": subj.get("definiteness"),
                        "pronominality": subj.get("pronominality"),
                    }
                    P = {
                        "animacy": obj.get("animacy"),
                        "definiteness": obj.get("definiteness"),
                        "pronominality": obj.get("pronominality"),
                    }

                    markA, markP = should_mark(rule, feature, direction, A, P)

                    subj_span = dict(subj)
                    obj_span = dict(obj)
                    subj_marked = dict(subj_span)
                    obj_marked = dict(obj_span)
                    subj_marked["text"] = f"{subj_span['text']} {AGENT_MARK}"
                    obj_marked["text"] = f"{obj_span['text']} {PATIENT_MARK}"

                    good_spans, bad_spans = [], []

                    if rule == "localA":
                        good_spans = [subj_marked] if markA else []
                        bad_spans = [] if markA else [subj_marked]
                    elif rule == "localP":
                        good_spans = [obj_marked] if markP else []
                        bad_spans = [] if markP else [obj_marked]
                    else:  # global: both-or-none
                        good_spans = [subj_marked, obj_marked] if (markA and markP) else []
                        bad_spans = [] if (markA and markP) else [subj_marked, obj_marked]

                    sentence_good = apply_spans(tokens, good_spans)
                    sentence_bad = apply_spans(tokens, bad_spans)

                    if strip_marks(sentence_good) != strip_marks(sentence_bad):
                        continue

                    item = {
                        "run_id": args.run_id,
                        "rule": rule,
                        "feature": feature,
                        "direction": direction,
                        "id": f"{data.get('index', 'na')}",
                        "sentence_good": sentence_good,
                        "sentence_bad": sentence_bad,
                        "type": "should_mark" if (good_spans and not bad_spans) else "should_not_mark",
                        "role": ("A" if rule == "localA" else ("P" if rule == "localP" else "both")),
                    }

                    if item["type"] == "should_mark":
                        should_pool.append(item)
                    else:
                        shouldnot_pool.append(item)

                    picked += 1

    half = args.num_pairs // 2
    random.shuffle(should_pool)
    random.shuffle(shouldnot_pool)
    pairs = should_pool[:half] + shouldnot_pool[:half]
    random.shuffle(pairs)

    out_jsonl = out_root / "test_minimal_pairs.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for x in pairs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    summary = {
        "run_id": args.run_id,
        "rule": rule,
        "feature": feature,
        "direction": direction,
        "inputs": inputs,
        "seed": args.seed,
        "requested": args.num_pairs,
        "written": len(pairs),
        "pool_should_mark": len(should_pool),
        "pool_should_not_mark": len(shouldnot_pool),
        "output": str(out_jsonl),
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
