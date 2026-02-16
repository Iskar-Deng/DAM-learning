#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_processing/extract_verb.py
Extract verb–argument structures (subjects, objects, clauses, PPs)
from parsed JSONL files.

Usage
------
# Default: process all *_parsed.jsonl files
python -m data_processing.extract_verb
"""

import os
import sys
import json
from collections import defaultdict
from tqdm import tqdm
from utils import DATA_PATH

def build_children_index(tokens):
    children = defaultdict(list)
    for t in tokens:
        hid = t.get("head", -1)
        if isinstance(hid, int):
            children[hid].append(t)
    return children

def extract_np_span(token_id, id_to_token, children):
    head = id_to_token[token_id]
    span = [head]
    for child in children.get(token_id, []):
        if child.get("dep") in {"det", "amod", "compound", "poss", "nmod"}:
            span.append(child)
    span = sorted(span, key=lambda t: t["id"])
    ids = [t["id"] for t in span]
    return {
        "text": " ".join(t["text"] for t in span),
        "span": [min(ids), max(ids)],
        "head": head["text"],
    }

def extract_clause_span(token_id, id_to_token, children):
    visited = set()
    stack = [token_id]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for ch in children.get(cur, []):
            cid = ch.get("id")
            if isinstance(cid, int) and cid not in visited:
                stack.append(cid)

    tokens = sorted((id_to_token[i] for i in visited if i in id_to_token),
                    key=lambda t: t["id"])
    if not tokens:
        head = id_to_token[token_id]["text"]
        return {"text": "", "span": [token_id, token_id], "head": head}

    ids = [t["id"] for t in tokens]
    return {
        "text": " ".join(t["text"] for t in tokens),
        "span": [min(ids), max(ids)],
        "head": id_to_token[token_id]["text"],
    }

def extract_verb_arguments(tokens):
    if not tokens:
        return []

    id_to_token = {t["id"]: t for t in tokens if isinstance(t.get("id"), int)}
    children = build_children_index(tokens)
    verbs = [t for t in tokens if t.get("pos") == "VERB"]
    results = []

    for verb in verbs:
        vid = verb["id"]
        out = {
            "verb": verb.get("lemma", verb.get("text", "")),
            "verb_id": vid,
            "subject": None,
            "objects": [],
        }

        for child in children.get(vid, []):
            dep = child.get("dep")
            tid = child.get("id")
            if not isinstance(tid, int):
                continue

            if dep in {"nsubj", "nsubjpass", "csubj"}:
                try:
                    out["subject"] = extract_np_span(tid, id_to_token, children)
                except Exception:
                    head = id_to_token[tid]["text"]
                    out["subject"] = {"text": "", "span": [tid, tid], "head": head}

            elif dep in {"dobj", "obj", "attr", "dative"}:
                try:
                    span = extract_np_span(tid, id_to_token, children)
                except Exception:
                    head = id_to_token[tid]["text"]
                    span = {"text": "", "span": [tid, tid], "head": head}
                span["dep"] = dep
                out["objects"].append(span)

            elif dep in {"xcomp", "ccomp"}:
                span = extract_clause_span(tid, id_to_token, children)
                span["dep"] = dep
                out["objects"].append(span)

            elif dep == "prep":
                prep = child
                subtree = [prep]

                for ch in children.get(prep["id"], []):
                    if ch.get("dep") == "pobj":
                        subtree.append(ch)
                        for gc in children.get(ch["id"], []):
                            if gc.get("dep") in {"det", "amod", "compound", "poss"}:
                                subtree.append(gc)

                if len(subtree) > 1:
                    toks = sorted((t for t in subtree if isinstance(t.get("id"), int)), key=lambda x: x["id"])
                    ids = [t["id"] for t in toks]
                    pobj_head = next((t["text"] for t in subtree if t.get("dep") == "pobj"), None)

                    out["objects"].append({
                        "text": " ".join(t["text"] for t in toks),
                        "span": [min(ids), max(ids)] if ids else [vid, vid],
                        "head": pobj_head,
                        "dep": "prep+pobj",
                    })

        results.append(out)

    return results

def process_file(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_lines = n_empty = n_errors = n_verbs = 0

    with open(input_path, encoding="utf-8") as fin, \
        open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"Extracting {os.path.basename(input_path)}"):
            if not line.strip():
                continue
            n_lines += 1

            try:
                data = json.loads(line)
            except Exception:
                n_errors += 1
                continue

            tokens = data.get("tokens") or []
            idx = data.get("index", -1)

            if not tokens:
                n_empty += 1
                fout.write(json.dumps({"index": idx, "tokens": [], "verbs": []}, ensure_ascii=False) + "\n")
                continue

            try:
                verbs = extract_verb_arguments(tokens)
            except Exception:
                n_errors += 1
                verbs = []

            n_verbs += len(verbs)
            fout.write(json.dumps({"index": idx, "tokens": tokens, "verbs": verbs}, ensure_ascii=False) + "\n")

    return {
        "lines": n_lines,
        "empty": n_empty,
        "errors": n_errors,
        "verbs_total": n_verbs,
        "out": output_path,
    }

def main():
    parsed_dir = os.path.join(DATA_PATH, "parsed")
    out_dir = os.path.join(DATA_PATH, "structured")
    os.makedirs(out_dir, exist_ok=True)

    jsonl_files = sorted([f for f in os.listdir(parsed_dir) if f.endswith("_parsed.jsonl")])

    if not jsonl_files:
        print(f"[ERROR] No *_parsed.jsonl in {parsed_dir}", file=sys.stderr)
        sys.exit(1)

    report = {}

    for fname in jsonl_files:
        src = os.path.join(parsed_dir, fname)
        dst = os.path.join(out_dir, fname.replace("_parsed.jsonl", "_verbs.jsonl"))

        stats = process_file(src, dst)
        key = fname.replace("_parsed.jsonl", "")
        report[key] = stats

        print(f"[OK] {dst} | lines={stats['lines']} empty={stats['empty']} "
            f"errors={stats['errors']} verbs={stats['verbs_total']}")

    print("\n[SUMMARY]")
    for k, s in report.items():
        print(f" - {k}: lines={s['lines']}, empty={s['empty']}, "
            f"errors={s['errors']}, verbs={s['verbs_total']}, out={s['out']}")

if __name__ == "__main__":
    main()
