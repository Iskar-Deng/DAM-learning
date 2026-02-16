#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
perturbation/run_perturb.py
Apply DAM markers to structured_labeled data using baseline/full/local-A, local-P, or global rules.

Usage
------
python -m perturbation.run_perturb --rule baseline
python -m perturbation.run_perturb --rule full
python -m perturbation.run_perturb --rule localA  --feature animacy       --direction natural
python -m perturbation.run_perturb --rule localP  --feature animacy       --direction inverse
python -m perturbation.run_perturb --rule global  --feature definiteness  --direction inverse

"""

import os
import json
import argparse
from glob import glob
from tqdm import tqdm

from utils import (
    DATA_PATH,
    AGENT_MARK,
    PATIENT_MARK,
    should_mark_local_A,
    should_mark_local_P,
    should_mark_global,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["baseline", "full", "localA", "localP", "global"], required=True)
    ap.add_argument("--feature", choices=["animacy", "definiteness", "pronominality"], default=None)
    ap.add_argument("--direction", choices=["natural", "inverse"], default=None)
    ap.add_argument("--input_dir", default=os.path.join(DATA_PATH, "structured_labeled"))
    ap.add_argument("--output_dir", default=os.path.join(DATA_PATH, "perturbed"))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_limit", type=int, default=200)
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


def apply_spans_to_tokens(tokens, spans):
    if not spans:
        return " ".join(t["text"] for t in tokens)

    seen = set()
    uniq = []
    for sp in spans:
        s, e = sp["span"]
        key = (s, e, sp["text"])
        if key not in seen:
            seen.add(key)
            uniq.append(sp)

    spans = sorted(uniq, key=lambda s: s["span"][0])
    out, i, n = [], 0, len(tokens)

    for sp in spans:
        s, e = sp["span"]
        if s < i:
            continue
        while i < s and i < n:
            out.append(tokens[i]["text"])
            i += 1
        out.append(sp["text"])
        i = max(i, e + 1)

    while i < n:
        out.append(tokens[i]["text"])
        i += 1

    return " ".join(out)


def process_one_jsonl(input_path, out_dir, rule, feature, direction, debug=False, debug_limit=200):
    fname = os.path.basename(input_path)
    prefix = fname.replace("_verbs_labeled.ok.jsonl", "")

    affected, unaffected, invalid_new = [], [], []
    debug_samples = []
    stats = {"affected": 0, "unaffected": 0, "invalid": 0}

    print(f"[INFO] Processing: {fname}")
    pbar = tqdm(desc=f"{fname}", unit="lines")

    with open(input_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin, 1):
            pbar.update(1)
            if debug and i > debug_limit:
                break
            if not line.strip():
                continue

            data = json.loads(line)
            tokens = data.get("tokens") or []
            sent_text = " ".join(t["text"] for t in tokens) if tokens else ""
            verbs = data.get("verbs") or []

            if not tokens or not verbs:
                invalid_new.append(sent_text)
                stats["invalid"] += 1
                continue

            spans = []
            has_valid = False
            has_marked = False

            for entry in verbs:
                if not is_valid_structure(entry):
                    continue
                has_valid = True

                subj = entry["subject"]
                obj = entry["objects"][0]

                A_labels = {
                    "animacy": subj.get("animacy"),
                    "definiteness": subj.get("definiteness"),
                    "pronominality": subj.get("pronominality"),
                }
                P_labels = {
                    "animacy": obj.get("animacy"),
                    "definiteness": obj.get("definiteness"),
                    "pronominality": obj.get("pronominality"),
                }

                if rule == "baseline":
                    markA, markP = False, False
                elif rule == "full":
                    markA, markP = True, True
                elif rule == "localA":
                    markA = should_mark_local_A(A_labels, feature, direction)
                    markP = False
                elif rule == "localP":
                    markA = False
                    markP = should_mark_local_P(P_labels, feature, direction)
                elif rule == "global":
                    markA, markP = should_mark_global(A_labels, P_labels, feature, direction)
                else:
                    raise ValueError(f"Unknown rule: {rule}")

                if markA:
                    s = dict(subj)
                    s["text"] = f"{s['text']} {AGENT_MARK}"
                    spans.append(s)
                    has_marked = True

                if markP:
                    o = dict(obj)
                    o["text"] = f"{o['text']} {PATIENT_MARK}"
                    spans.append(o)
                    has_marked = True

            if has_valid and has_marked:
                perturbed = apply_spans_to_tokens(tokens, spans)
                affected.append(perturbed)
                stats["affected"] += 1
                if debug and len(debug_samples) < 20:
                    debug_samples.append(
                        {"original": sent_text, "perturbed": perturbed, "rule": rule, "feature": feature, "direction": direction}
                    )
            elif has_valid:
                unaffected.append(sent_text)
                stats["unaffected"] += 1
            else:
                invalid_new.append(sent_text)
                stats["invalid"] += 1

    pbar.close()

    with open(os.path.join(out_dir, f"{prefix}_affected.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(affected) + ("\n" if affected else ""))
    with open(os.path.join(out_dir, f"{prefix}_unaffected.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(unaffected) + ("\n" if unaffected else ""))
    with open(os.path.join(out_dir, f"{prefix}_invalid.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(invalid_new) + ("\n" if invalid_new else ""))

    if debug and debug_samples:
        dbg_path = os.path.join(out_dir, f"{prefix}_debug_preview.jsonl")
        with open(dbg_path, "w", encoding="utf-8") as f:
            for x in debug_samples:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[DEBUG] Preview → {dbg_path}")

    return {"input": input_path, "stats": stats}


def main():
    args = parse_args()

    if args.rule in ("baseline", "full"):
        feature, direction = "N/A", "N/A"
        out_dir = os.path.join(args.output_dir, args.rule)
    else:
        if args.feature is None or args.direction is None:
            raise ValueError("--feature and --direction are required unless --rule is baseline/full.")
        feature, direction = args.feature, args.direction
        out_dir = os.path.join(args.output_dir, f"{args.rule}_{feature}_{direction}")

    inputs = sorted(glob(os.path.join(args.input_dir, "*_verbs_labeled.ok.jsonl")))
    if not inputs:
        raise FileNotFoundError("No *_verbs_labeled.ok.jsonl found.")

    os.makedirs(out_dir, exist_ok=True)

    summaries = [
        process_one_jsonl(
            ip,
            out_dir,
            rule=args.rule,
            feature=feature,
            direction=direction,
            debug=args.debug,
            debug_limit=args.debug_limit,
        )
        for ip in inputs
    ]

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    totals = {
        "affected": sum(s["stats"]["affected"] for s in summaries),
        "unaffected": sum(s["stats"]["unaffected"] for s in summaries),
        "invalid": sum(s["stats"]["invalid"] for s in summaries),
    }

    with open(os.path.join(out_dir, "aggregate_stats.json"), "w", encoding="utf-8") as f:
        json.dump(totals, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Saved → {out_dir}")
    print(f"Totals: {totals}")


if __name__ == "__main__":
    main()
