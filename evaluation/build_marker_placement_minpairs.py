#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/build_marker_placement_minpairs.py
Build minimal pairs for marker placement robustness (localA / localP / global).

Reads:
  $DATA_PATH/structured_labeled/*_{split}_verbs_labeled.ok.jsonl

Outputs:
  $EVALUATION_PATH/marker_placement/minimal_pairs/<run_id>/test_marker_position_minpairs.jsonl
  $EVALUATION_PATH/marker_placement/minimal_pairs/<run_id>/summary.json

Usage
------
python -m evaluation.build_marker_placement_minpairs --run_id localA_animacy_natural
python -m evaluation.build_marker_placement_minpairs --run_id global_definiteness_inverse --num_pairs 2000
"""

import os
import json
import random
import argparse
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input_dir", default=os.path.join(DATA_PATH, "structured_labeled"))
    ap.add_argument("--out_dir", default=os.path.join(EVALUATION_PATH, "marker_placement", "minimal_pairs"))
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--max_per_sentence", type=int, default=1)
    return ap.parse_args()


def parse_run_id(run_id: str) -> Tuple[str, Optional[str], Optional[str]]:
    parts = run_id.split("_")
    rule = parts[0]
    if rule not in {"localA", "localP", "global"}:
        raise ValueError(f"run_id must start with localA/localP/global, got: {run_id}")
    if len(parts) < 3:
        raise ValueError(f"run_id must look like {rule}_<feature>_<direction>")
    feature, direction = parts[1], parts[2]
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature: {feature}")
    if direction not in {"natural", "inverse"}:
        raise ValueError(f"Unknown direction: {direction}")
    return rule, feature, direction


def is_valid_structure(entry: dict) -> bool:
    if not entry.get("subject"):
        return False
    objs = entry.get("objects") or []
    if len(objs) != 1:
        return False
    if objs[0].get("dep") == "ccomp":
        return False
    return True


def strip_marks(s: str) -> str:
    return " ".join(s.replace(AGENT_MARK, "").replace(PATIENT_MARK, "").split()).strip()


def iter_inputs(input_dir: str, splits: List[str]) -> List[str]:
    paths = []
    for sp in splits:
        paths.extend(sorted(glob(os.path.join(input_dir, f"*_{sp}_verbs_labeled.ok.jsonl"))))
    if not paths:
        raise FileNotFoundError(f"No *_<split>_verbs_labeled.ok.jsonl under {input_dir} for splits={splits}")
    return paths


def should_mark(rule: str, feature: str, direction: str, A: Dict[str, str], P: Dict[str, str]) -> Tuple[bool, bool]:
    if rule == "localA":
        return should_mark_local_A(A, feature, direction), False
    if rule == "localP":
        return False, should_mark_local_P(P, feature, direction)
    if rule == "global":
        return should_mark_global(A, P, feature, direction)
    raise ValueError(rule)


def build_with_insertions(tokens: List[dict], insert_after: Dict[int, List[str]]) -> str:
    out = []
    for i, tok in enumerate(tokens):
        out.append(tok.get("text", ""))
        if i in insert_after:
            out.extend(insert_after[i])
    return " ".join(out)


def pick_shift(base_idx: int, n: int) -> Tuple[int, Dict]:
    cands = [-2, -1, 1, 2]
    random.shuffle(cands)
    for d in cands:
        j = base_idx + d
        if 0 <= j < n:
            return j, {"direction": "left" if d < 0 else "right", "delta": abs(d)}
    return -1, {"direction": "none", "delta": 0}


def main():
    args = parse_args()
    random.seed(args.seed)

    rule, feature, direction = parse_run_id(args.run_id)
    inputs = iter_inputs(args.input_dir, args.splits)

    out_root = Path(args.out_dir) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_root / "test_marker_position_minpairs.jsonl"

    pairs = []

    for ip in inputs:
        with open(ip, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc=os.path.basename(ip)):
                if len(pairs) >= args.num_pairs:
                    break
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                tokens = data.get("tokens") or []
                verbs = data.get("verbs") or []
                if not tokens or not verbs:
                    continue

                picked = 0
                for entry in verbs:
                    if len(pairs) >= args.num_pairs or picked >= args.max_per_sentence:
                        break
                    if not is_valid_structure(entry):
                        continue

                    subj = entry["subject"]
                    obj = entry["objects"][0]
                    sA, eA = subj.get("span", [None, None])
                    sP, eP = obj.get("span", [None, None])
                    if sA is None or eA is None or sP is None or eP is None:
                        continue

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

                    if rule in {"localA", "localP"}:
                        if not (markA ^ markP):
                            continue
                    else:
                        if not (markA and markP):
                            continue

                    roles = []
                    if markA:
                        roles.append(("A", int(eA), AGENT_MARK))
                    if markP:
                        roles.append(("P", int(eP), PATIENT_MARK))
                    if not roles:
                        continue

                    role_to_move, base_idx, _ = random.choice(roles)
                    new_idx, shift = pick_shift(base_idx, len(tokens))
                    if new_idx < 0 or new_idx == base_idx:
                        continue

                    ins_good: Dict[int, List[str]] = {}
                    for r, idx, m in roles:
                        ins_good.setdefault(idx, []).append(m)

                    ins_bad: Dict[int, List[str]] = {}
                    for r, idx, m in roles:
                        ins_bad.setdefault(new_idx if r == role_to_move else idx, []).append(m)

                    sentence_good = build_with_insertions(tokens, ins_good)
                    sentence_bad = build_with_insertions(tokens, ins_bad)

                    if strip_marks(sentence_good) != strip_marks(sentence_bad):
                        continue

                    pairs.append(
                        {
                            "run_id": args.run_id,
                            "rule": rule,
                            "feature": feature,
                            "direction": direction,
                            "id": f"{data.get('index', 'na')}",
                            "sentence_good": sentence_good,
                            "sentence_bad": sentence_bad,
                            "type": "misplaced_marker",
                            "role": role_to_move,
                            "shift": shift,
                        }
                    )
                    picked += 1

    with out_jsonl.open("w", encoding="utf-8") as f:
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
        "output": str(out_jsonl),
        "note": "Each pair moves exactly one required marker by 1–2 tokens; others (if any) stay at correct NP end.",
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
