#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/run_probing.py
Run semantic probing on one checkpoint and one probe dataset (subject/object heads).

Usage
------
python -m evaluation.run_probing --ckpt /path/to/checkpoint-15000 --data /path/to/probe_subject_heads_test.jsonl --out /path/to/out.json
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    f1s = []
    for c in (0, 1):
        tp = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != c and yp == c) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == c and yp != c) for yt, yp in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(sum(f1s) / 2)


def feature_to_binary(feat: str, value: str) -> Optional[int]:
    v = (value or "").strip().lower()
    if feat == "animacy":
        return 0 if v == "inanimate" else (1 if v == "animate" else None)
    if feat == "definiteness":
        return 0 if v == "indef" else (1 if v == "definite" else None)
    if feat == "pronominality":
        return 0 if v == "common" else (1 if v == "pronoun" else None)
    raise ValueError(feat)


def type_based_split(
    y: List[int],
    head_words: List[str],
    test_type_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], Dict[int, int], Dict[int, int]]:
    rng = random.Random(seed)
    types_by_c = {0: set(), 1: set()}
    for yi, w in zip(y, head_words):
        types_by_c[yi].add(w)

    test_types = {0: set(), 1: set()}
    train_types = {0: set(), 1: set()}

    for c in (0, 1):
        types = list(types_by_c[c])
        rng.shuffle(types)
        n_test = int(math.ceil(len(types) * test_type_ratio))
        test_types[c] = set(types[:n_test])
        train_types[c] = set(types[n_test:])

    tr, te = [], []
    for i, (yi, w) in enumerate(zip(y, head_words)):
        (te if w in test_types[yi] else tr).append(i)

    return tr, te, {0: len(train_types[0]), 1: len(train_types[1])}, {0: len(test_types[0]), 1: len(test_types[1])}


def subsample_per_class(idxs: List[int], y: List[int], n_per_class: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    byc = {0: [], 1: []}
    for i in idxs:
        byc[y[i]].append(i)
    for c in (0, 1):
        rng.shuffle(byc[c])
        byc[c] = byc[c][:n_per_class]
    out = byc[0] + byc[1]
    rng.shuffle(out)
    return out


@torch.inference_mode()
def extract_embeddings(
    model,
    tokenizer,
    items: List[Dict[str, Any]],
    max_length: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    model.eval()
    H = int(model.config.hidden_size)
    embs = torch.empty((len(items), H), dtype=torch.float32)

    for start in tqdm(range(0, len(items), batch_size), desc="Extract", leave=False):
        batch = items[start:start + batch_size]
        texts = [b["sentence"] for b in batch]
        spans = [b["head_char_span"] for b in batch]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping").to(device)
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True, return_dict=True)
        last = out.hidden_states[-1]

        for bi in range(last.size(0)):
            s, e = spans[bi]
            attn = enc["attention_mask"][bi]
            T = int(attn.sum().item())
            tok_offsets = offsets[bi]

            chosen = None
            for ti in range(T):
                a, b = tok_offsets[ti].tolist()
                if a == 0 and b == 0:
                    continue
                if not (b <= s or a >= e):
                    chosen = ti
            if chosen is None:
                chosen = max(T - 1, 0)

            embs[start + bi] = last[bi, chosen].detach().float().cpu()

    return embs


class LinearProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 2)

    def forward(self, x):
        return self.fc(x)


def train_probe(Xtr: torch.Tensor, ytr: torch.Tensor, seed: int) -> LinearProbe:
    torch.manual_seed(seed)
    m = LinearProbe(Xtr.size(1))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.0)
    m.train()
    for _ in range(200):
        logits = m(Xtr)
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


@torch.inference_mode()
def eval_probe(m: LinearProbe, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    m.eval()
    pred = m(X).argmax(dim=-1).cpu().tolist()
    gold = y.cpu().tolist()
    acc = sum(int(p == g) for p, g in zip(pred, gold)) / len(gold) if gold else float("nan")
    f1 = macro_f1(gold, pred) if gold else float("nan")
    return float(acc), float(f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_type_ratio", type=float, default=0.2)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = Path(args.ckpt)
    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(ckpt), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(str(ckpt))
    model.to(device)
    model.eval()

    items = load_jsonl(data_path)
    X = extract_embeddings(model, tok, items, args.max_length, args.batch_size, device)

    head_words_all = [str(it.get("head_word", "")).lower().strip() for it in items]

    feats = {}
    for feat_name in ("animacy", "definiteness", "pronominality"):
        y_raw = [feature_to_binary(feat_name, (it.get("features", {}) or {}).get(feat_name, "")) for it in items]
        keep = [i for i, yi in enumerate(y_raw) if yi is not None and head_words_all[i]]
        if len(keep) < 10:
            feats[feat_name] = {"acc": float("nan"), "macro_f1": float("nan")}
            continue

        y = [y_raw[i] for i in keep]
        words = [head_words_all[i] for i in keep]
        Xk = X[keep]

        tr_idx, te_idx, n_tr_types, n_te_types = type_based_split(y, words, args.test_type_ratio, args.seed)
        tr_idx = subsample_per_class(tr_idx, y, args.n_per_class, seed=args.seed + 13)
        te_idx = subsample_per_class(te_idx, y, args.n_per_class, seed=args.seed + 17)

        Xtr = Xk[tr_idx].float()
        ytr = torch.tensor([y[i] for i in tr_idx], dtype=torch.long)
        Xte = Xk[te_idx].float()
        yte = torch.tensor([y[i] for i in te_idx], dtype=torch.long)

        cc = Counter(ytr.tolist())
        probe = train_probe(Xtr, ytr, seed=args.seed)
        acc, f1 = eval_probe(probe, Xte, yte)

        feats[feat_name] = {
            "acc": acc,
            "macro_f1": f1,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "n_train_types": int(n_tr_types[0] + n_tr_types[1]),
            "n_test_types": int(n_te_types[0] + n_te_types[1]),
            "class_counts_kept": {"neg(0)": int(cc.get(0, 0)), "pos(1)": int(cc.get(1, 0))},
        }

    result = {
        "ckpt": str(ckpt),
        "data": str(data_path),
        "seed": args.seed,
        "n_per_class": args.n_per_class,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "test_type_ratio": args.test_type_ratio,
        "features": feats,
    }

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
