#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_processing/parse.py
Constituency parsing using spaCy + benepar (batch mode, resume supported).

Usage
------
# Default settings (sm model, batch_size=32)
python -m data_processing.parse

# Custom batch size
python -m data_processing.parse --batch-size 16

# Resume from previous partial output
python -m data_processing.parse --resume
"""

import os
import sys
import json
import argparse
import pathlib
from typing import Iterator, Tuple, List

import spacy
import benepar
from tqdm import tqdm
from utils import DATA_PATH

def parse_args():
    ap = argparse.ArgumentParser(description="Constituency parsing for filtered corpus.")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for spaCy pipe.")
    ap.add_argument("--n-process", type=int, default=1, help="spaCy n_process (benepar prefers 1).")
    ap.add_argument("--resume", action="store_true", help="Resume from existing parsed outputs.")
    return ap.parse_args()

filtered_dir = os.path.join(DATA_PATH, "filtered")
parsed_dir = os.path.join(DATA_PATH, "parsed")
os.makedirs(parsed_dir, exist_ok=True)

txt_files = sorted([f for f in os.listdir(filtered_dir) if f.endswith(".txt")])
if not txt_files:
    print(f"[ERROR] No .txt files found in {filtered_dir}", file=sys.stderr)
    sys.exit(1)

def build_nlp():
    """Load spaCy small model + benepar parser."""
    model_name = "en_core_web_sm"

    try:
        nlp = spacy.load(model_name, disable=["ner"])
    except OSError:
        print(f"[INFO] Missing spaCy model {model_name}. Install via:", file=sys.stderr)
        print(f"       python -m spacy download {model_name}", file=sys.stderr)
        sys.exit(1)

    try:
        benepar.load_trained_model("benepar_en3")
    except Exception:
        print("[INFO] Downloading benepar_en3...", file=sys.stderr)
        benepar.download("benepar_en3")

    if "benepar" not in nlp.pipe_names:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"}, last=True)

    nlp.max_length = 5_000_000
    return nlp

def annotate_doc(doc, index: int):
    """Convert a parsed spaCy doc to a JSON-serializable object."""
    try:
        sent = list(doc.sents)[0]
    except Exception:
        return {"index": index, "tokens": [], "ptb": "", "error": "no_sentence"}

    tokens = [
        {
            "id": tok.i,
            "text": tok.text,
            "lemma": tok.lemma_,
            "pos": tok.pos_,
            "dep": tok.dep_,
            "head": tok.head.i,
            "ner": tok.ent_type_ or "O",
        }
        for tok in sent
    ]

    ptb = getattr(sent._, "parse_string", "")
    return {"index": index, "tokens": tokens, "ptb": ptb}

def clean_text(s: str):
    return s.replace("\u200b", "").replace("\u200e", "").replace("\ufeff", "").strip()

def count_nonempty_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def get_resume_index(jsonl_path: str) -> int:
    """Find last index in parsed JSONL, or -1 if none."""
    try:
        if not os.path.exists(jsonl_path) or os.path.getsize(jsonl_path) == 0:
            return -1

        with open(jsonl_path, "rb") as f:
            f.seek(0, 2)
            size = pos = f.tell()
            chunk = 1024
            buf = b""

            while pos > 0:
                pos = max(0, pos - chunk)
                f.seek(pos)
                buf = f.read(size - pos) + buf
                if b"\n" in buf:
                    break

        last = buf.splitlines()[-1].decode("utf-8", "ignore").strip()
        if not last:
            return -1

        obj = json.loads(last)
        return int(obj.get("index", -1))

    except Exception:
        return -1

def nonempty_texts_and_idxs(path: str, start_idx: int) -> Iterator[Tuple[str, int]]:
    idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if idx >= start_idx:
                yield s, idx
            idx += 1

def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def main(batch_size=32, n_process=1, resume=False):
    nlp = build_nlp()

    for name in txt_files:
        src = os.path.join(filtered_dir, name)
        dst = os.path.join(parsed_dir, name.replace(".txt", "_parsed.jsonl"))
        bad = os.path.join(parsed_dir, name.replace(".txt", "_badlines.log"))

        total_nonempty = count_nonempty_lines(src)

        resume_from = get_resume_index(dst) if resume else -1
        start_idx = resume_from + 1
        remaining = total_nonempty - start_idx

        print(
            f"[INFO] Parsing {name} → {pathlib.Path(dst).name} "
            f"(nonempty={total_nonempty}, start={start_idx}, remaining={remaining})"
        )

        fout_mode = "a" if resume_from >= 0 else "w"

        with open(dst, fout_mode, encoding="utf-8") as fout, \
            open(bad, fout_mode, encoding="utf-8") as fbad:

            pbar = tqdm(total=total_nonempty, initial=start_idx, desc=f"Parsing {name}",
                        mininterval=60, dynamic_ncols=True)

            stream = nonempty_texts_and_idxs(src, start_idx)

            for batch in batched(stream, batch_size):
                texts = [t for t, _ in batch]
                idxs = [i for _, i in batch]

                try:
                    docs = list(nlp.pipe(texts, batch_size=batch_size, n_process=1))

                    for doc, i in zip(docs, idxs):
                        try:
                            parsed = annotate_doc(doc, i)
                        except Exception as e:
                            fbad.write(f"{i}\t{type(e).__name__}: {e}\n")
                            parsed = {"index": i, "tokens": [], "ptb": "", "error": str(e)}

                        fout.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                        pbar.update(1)

                except Exception:
                    # fallback per-line
                    for text, i in batch:
                        try:
                            doc = nlp(text)
                        except Exception:
                            doc = nlp(clean_text(text))

                        try:
                            parsed = annotate_doc(doc, i)
                        except Exception as e:
                            fbad.write(f"{i}\t{type(e).__name__}: {e}\n")
                            parsed = {"index": i, "tokens": [], "ptb": "", "error": str(e)}

                        fout.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                        pbar.update(1)

            pbar.close()

        print(f"[DONE] {dst}  (bad lines logged → {pathlib.Path(bad).name})")

if __name__ == "__main__":
    args = parse_args()
    main(batch_size=args.batch_size, n_process=args.n_process, resume=args.resume)
