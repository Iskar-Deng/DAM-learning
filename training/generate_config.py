#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training/generate_config.py
Generate training YAML config (clean version).

- Merge short lines for I/O efficiency
- Optional static oversample
- Use defaults from utils.py
- Fixed-step training (no epoch logic)
"""

import argparse
import yaml
import math
import random
from pathlib import Path

from utils import (
    CONFIG_PATH,
    CHECKPOINT_PATH,
    CACHE_PATH,
    MODEL_NAME,
    SEED,
    BLOCK_SIZE,
    TRAINING_ARGUMENTS_DEFAULTS,
    CHECKPOINT_FREQUENCY_DEFAULTS,
    RESUME_DEFAULT,
)


# =====================================================
# Helpers
# =====================================================

def merge_short_lines(path: Path, max_chars: int = 1000) -> Path:
    merged_path = path.with_name(f"{path.stem}_merged.txt")
    with open(path, "r", encoding="utf-8") as f, open(merged_path, "w", encoding="utf-8") as out:
        buf, length = [], 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if length + len(line) > max_chars:
                out.write(" ".join(buf) + "\n")
                buf, length = [line], len(line)
            else:
                buf.append(line)
                length += len(line)
        if buf:
            out.write(" ".join(buf) + "\n")
    print(f"[Merge] {path.name} → {merged_path.name}")
    return merged_path


def static_oversample(path: Path, ratio: float, seed: int = 42) -> Path:
    if ratio <= 1.0:
        return path

    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip()]

    if not lines:
        return path

    n = len(lines)
    n_full = int(ratio)
    frac = ratio - n_full

    oversampled = lines * n_full
    if frac > 1e-6:
        random.seed(seed)
        n_take = max(1, int(math.ceil(n * frac)))
        oversampled.extend(random.sample(lines, n_take))

    out_path = path.with_name(f"{path.stem}_x{ratio:.2f}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for line in oversampled:
            f.write(line + "\n")

    print(f"[Oversample] {path.name}: {n} → {len(oversampled)} lines ({ratio}x)")
    return out_path

def build_cfg(base_dir: Path, oversample: float):
    run_id = base_dir.name

    def find_paths(split: str):
        def one(kind: str):
            matches = list(base_dir.glob(f"*_{split}_{kind}.txt"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one '*_{split}_{kind}.txt' in {base_dir}, "
                    f"found {len(matches)}"
                )
            return matches[0]

        return {
            "affected": one("affected"),
            "unaffected": one("unaffected"),
            "invalid": one("invalid"),
        }

    train_raw = find_paths("train")
    valid_raw = find_paths("valid")

    train_merged = {k: merge_short_lines(v) for k, v in train_raw.items()}
    valid_merged = {k: merge_short_lines(v) for k, v in valid_raw.items()}

    train_final = {
        "affected": static_oversample(train_merged["affected"], oversample),
        "unaffected": static_oversample(train_merged["unaffected"], oversample),
        "invalid": train_merged["invalid"],
    }

    cfg = {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "seed": SEED,
        "block_size": BLOCK_SIZE,
        "resume": RESUME_DEFAULT,
        "resume_checkpoint": None,
        "checkpoint_frequency": CHECKPOINT_FREQUENCY_DEFAULTS,
        "artifacts": {
            "cache_dir": str(Path(CACHE_PATH) / run_id),
            "run_dir": str(Path(CHECKPOINT_PATH) / run_id),
        },
        "training_arguments": dict(TRAINING_ARGUMENTS_DEFAULTS),
        "data": {
            "train_files": [
                str(train_final["affected"]),
                str(train_final["unaffected"]),
                str(train_final["invalid"]),
            ],
            "eval_files": [
                str(valid_merged["affected"]),
                str(valid_merged["unaffected"]),
                str(valid_merged["invalid"]),
            ],
        },
    }

    eff_bsz = (
        cfg["training_arguments"]["per_device_train_batch_size"]
        * cfg["training_arguments"]["gradient_accumulation_steps"]
    )
    print(f"[Config] effective batch = {eff_bsz} seq/step")

    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="perturbed condition directory")
    ap.add_argument("--oversample", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.input_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(base_dir)

    cfg = build_cfg(base_dir, args.oversample)

    out_dir = Path(CONFIG_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / f"{cfg['run_id']}.yaml"

    if out_yaml.exists() and not args.overwrite:
        raise FileExistsError(f"{out_yaml} exists (use --overwrite)")

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"\n[OK] Wrote config → {out_yaml}")


if __name__ == "__main__":
    main()
