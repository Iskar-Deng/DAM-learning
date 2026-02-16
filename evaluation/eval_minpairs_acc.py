#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/eval_minpairs_acc.py
Evaluate minimal-pair accuracy on a single checkpoint.

Usage
------
python -m evaluation.eval_minpairs_acc \
  --checkpoint /path/to/checkpoint-8000 \
  --minpairs /path/to/valid_test_minimal_pairs.jsonl
"""

import argparse
import contextlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--minpairs", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args()


def read_minipairs(jsonl_path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            g = obj.get("sentence_good")
            b = obj.get("sentence_bad")
            if isinstance(g, str) and isinstance(b, str) and g and b:
                pairs.append((g, b))
    return pairs


def load_tokenizer(checkpoint: Path):
    tok = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(checkpoint: Path, device: str, torch_dtype: Optional[torch.dtype]):
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model


def batch_iter(items: List[Tuple[str, str]], batch_size: int) -> Iterable[List[Tuple[str, str]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _masked_labels(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    return labels


@torch.inference_mode()
def per_sample_mean_nll(texts: List[str], model, tokenizer, device: str) -> List[float]:
    """
    Returns length-normalized negative log-likelihood (mean NLL) per sample:
      mean over non-pad target tokens of CE(logits_t, token_{t+1}).
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=getattr(model.config, "n_positions", tokenizer.model_max_length),
    ).to(device)

    labels = _masked_labels(enc)
    outputs = model(**enc, labels=labels)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab = shift_logits.size(-1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_nll = loss_fct(shift_logits.view(-1, vocab), shift_labels.view(-1))
    token_nll = token_nll.view(shift_labels.size())

    mask = (shift_labels != -100).float()
    denom = mask.sum(dim=-1).clamp_min(1.0)
    mean_nll = (token_nll * mask).sum(dim=-1) / denom

    return mean_nll.detach().float().cpu().tolist()


def main():
    args = parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    minpairs_path = args.minpairs.resolve()
    if not minpairs_path.exists():
        raise FileNotFoundError(minpairs_path)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch_dtype = None
    if device == "cuda" and args.bf16:
        torch_dtype = torch.bfloat16
    elif device == "cuda" and args.fp16:
        torch_dtype = torch.float16

    tok = load_tokenizer(checkpoint)
    model = load_model(checkpoint, device=device, torch_dtype=torch_dtype)

    pairs = read_minipairs(minpairs_path)
    if not pairs:
        raise RuntimeError(f"No valid pairs in {minpairs_path}")

    correct = 0
    total = 0

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (args.fp16 and device == "cuda")
        else contextlib.nullcontext()
    )

    with amp_ctx:
        for batch in tqdm(
            batch_iter(pairs, args.batch_size),
            total=(len(pairs) + args.batch_size - 1) // args.batch_size,
            desc=f"Eval {checkpoint.name}",
        ):
            flat: List[str] = []
            for g, b in batch:
                flat.extend([g, b])

            nlls = per_sample_mean_nll(flat, model, tok, device=device)
            for i in range(len(batch)):
                ng = nlls[2 * i]
                nb = nlls[2 * i + 1]
                correct += int(ng < nb)
                total += 1

    acc = (correct / total) if total else float("nan")
    out = {
        "checkpoint": str(checkpoint),
        "minpairs": str(minpairs_path),
        "total": int(total),
        "correct": int(correct),
        "accuracy": float(acc),
        "device": device,
        "torch_dtype": str(torch_dtype) if torch_dtype is not None else None,
        "score": "length-normalized mean NLL (negative log-likelihood)",
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
