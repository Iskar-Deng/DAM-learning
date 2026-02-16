#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/build_probe_heads.py
Build semantic probing datasets (A/P head positions).

Reads:
  $DATA_PATH/structured_labeled/*_<split>_verbs_labeled.ok.jsonl

Writes:
  $EVALUATION_PATH/probing/data/
    probe_subject_heads_<split>.jsonl
    probe_object_heads_<split>.jsonl

Usage
------
python -m evaluation.build_probe_heads
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils import DATA_PATH, EVALUATION_PATH

PUNCT_NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "\"", "''", "``"}
CLITICS_NO_SPACE_BEFORE = {"n't", "'s", "'re", "'ve", "'m", "'d", "'ll"}
NO_SPACE_AFTER = {"(", "[", "{", "``"}


def detokenize_with_word_spans(tokens: List[Dict[str, Any]]) -> Tuple[str, Dict[int, Tuple[int, int]]]:
    text_parts = []
    spans: Dict[int, Tuple[int, int]] = {}

    cur_len = 0
    prev_tok = None

    for t in tokens:
        wid = int(t["id"])
        w = str(t["text"])

        need_space = True
        if cur_len == 0:
            need_space = False
        elif w in PUNCT_NO_SPACE_BEFORE or w in CLITICS_NO_SPACE_BEFORE:
            need_space = False
        elif prev_tok in NO_SPACE_AFTER:
            need_space = False

        if need_space:
            text_parts.append(" ")
            cur_len += 1

        start = cur_len
        text_parts.append(w)
        cur_len += len(w)
        end = cur_len

        spans[wid] = (start, end)
        prev_tok = w

    return "".join(text_parts), spans

def span_to_head_word_id(span: List[int], tokens_by_id: Dict[int, Dict[str, Any]], head_str: str) -> int:
    s, e = int(span[0]), int(span[1])
    hs = head_str.lower() if head_str else None

    if hs:
        for wid in range(e, s - 1, -1):
            if wid in tokens_by_id and tokens_by_id[wid]["text"].lower() == hs:
                return wid
    return e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "valid", "train"])
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path(DATA_PATH) / "structured_labeled",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(EVALUATION_PATH) / "probing" / "data",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"*_{args.split}_verbs_labeled.ok.jsonl"
    cands = sorted(args.data_dir.glob(pattern))
    if len(cands) != 1:
        raise RuntimeError(
            f"Expected exactly 1 file matching {pattern} under {args.data_dir}, "
            f"found {len(cands)}: {[str(p) for p in cands]}"
        )
    in_path = cands[0]

    out_subj = args.out_dir / f"probe_subject_heads_{args.split}.jsonl"
    out_obj = args.out_dir / f"probe_object_heads_{args.split}.jsonl"

    n_subj = n_obj = n_skipped = 0

    with in_path.open("r", encoding="utf-8") as f_in, \
            out_subj.open("w", encoding="utf-8") as f_subj, \
            out_obj.open("w", encoding="utf-8") as f_obj:

        for line in f_in:
            if not line.strip():
                continue

            obj = json.loads(line)
            tokens = obj.get("tokens") or []
            verbs = obj.get("verbs") or []
            if not tokens or not verbs:
                continue

            tokens_by_id = {int(t["id"]): t for t in tokens}
            sentence, word_spans = detokenize_with_word_spans(tokens)

            for fi, vf in enumerate(verbs):

                subj = vf.get("subject")
                objs = vf.get("objects") or []
                if not subj or len(objs) != 1:
                    continue
                obj_np = objs[0]
                if obj_np.get("dep") == "ccomp":
                    continue

                verb_lemma = str(vf.get("verb", ""))

                if "span" in subj:
                    head_id = span_to_head_word_id(subj["span"], tokens_by_id, subj.get("head"))
                    if head_id in word_spans:
                        sample = {
                            "sentence": sentence,
                            "head_word": tokens_by_id[head_id]["text"],
                            "head_word_id": head_id,
                            "head_char_span": list(word_spans[head_id]),
                            "role": "subj",
                            "verb_lemma": verb_lemma,
                            "frame_index": fi,
                            "features": {
                                "animacy": subj.get("animacy"),
                                "definiteness": subj.get("definiteness"),
                                "pronominality": subj.get("pronominality"),
                            },
                        }
                        f_subj.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        n_subj += 1
                    else:
                        n_skipped += 1

                # ----- object -----
                if "span" in obj_np:
                    head_id = span_to_head_word_id(obj_np["span"], tokens_by_id, obj_np.get("head"))
                    if head_id in word_spans:
                        sample = {
                            "sentence": sentence,
                            "head_word": tokens_by_id[head_id]["text"],
                            "head_word_id": head_id,
                            "head_char_span": list(word_spans[head_id]),
                            "role": "obj",
                            "verb_lemma": verb_lemma,
                            "frame_index": fi,
                            "features": {
                                "animacy": obj_np.get("animacy"),
                                "definiteness": obj_np.get("definiteness"),
                                "pronominality": obj_np.get("pronominality"),
                            },
                        }
                        f_obj.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        n_obj += 1
                    else:
                        n_skipped += 1

    print(f"[Input] {in_path}")
    print(f"[{args.split}] subj samples: {n_subj}")
    print(f"[{args.split}] obj  samples: {n_obj}")
    print(f"[{args.split}] skipped: {n_skipped}")
    print(f"[DONE] wrote -> {out_subj}")
    print(f"[DONE] wrote -> {out_obj}")


if __name__ == "__main__":
    main()
