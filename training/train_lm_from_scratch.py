#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training/train_lm_from_scratch.py
Train a causal LM from a YAML config file (clean version).

- Train GPT-2 from scratch (random init; architecture from model_name config)
- Eval enabled (uses eval_files in YAML)
- Resume default false; if resume true, must provide resume_checkpoint explicitly
- Cache/Temp redirected to YAML artifacts.cache_dir (no hard-coded paths)
"""

import os
import math
import time
import argparse
import tempfile
import random
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
import torch
import numpy
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    GPT2Config,
    GPT2LMHeadModel,
)

from utils import AGENT_MARK, PATIENT_MARK


def _expand(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


def _load_and_oversample(
    paths: List[str],
    oversample_plan: Optional[Dict[str, float]] = None,
    keep_tmp: bool = False,
):
    all_paths = []
    tmp_files = []

    if oversample_plan is None:
        return paths

    for p in paths:
        p_str = str(p)
        ratio = 1.0
        if "affected" in p_str:
            ratio = oversample_plan.get("affected", 1.0)
        elif "unaffected" in p_str:
            ratio = oversample_plan.get("unaffected", 1.0)

        n_full = int(ratio)
        frac = ratio - n_full
        all_paths.extend([p_str] * n_full)

        if frac > 1e-6:
            with open(p_str, "r", encoding="utf-8") as f:
                lines = [x.strip() for x in f if x.strip()]
            n_take = max(1, int(math.ceil(len(lines) * frac)))
            random.seed(42)
            sampled = random.sample(lines, n_take)

            tmpfile = tempfile.NamedTemporaryFile(
                mode="w", delete=False, encoding="utf-8", suffix="_partial.txt"
            )
            for line in sampled:
                tmpfile.write(line + "\n")
            tmpfile.close()

            tmp_files.append(tmpfile.name)
            print(f"[Info] fractional oversample: {p_str} +{frac:.2f}x ({n_take} lines) → {tmpfile.name}")
            all_paths.append(tmpfile.name)

    if not keep_tmp:
        import atexit

        def _cleanup():
            for f in tmp_files:
                try:
                    os.remove(f)
                except Exception:
                    pass

        atexit.register(_cleanup)

    return all_paths


def _build_dataset(files: List[str]):
    if not files:
        return None

    ds_list = []
    for f in files:
        f = _expand(f)
        if f.endswith(".txt"):
            ds = load_dataset("text", data_files=f, split="train")
        elif f.endswith(".jsonl") or f.endswith(".json"):
            ds = load_dataset("json", data_files=f, split="train")
            if "text" not in ds.column_names:
                raise ValueError(f"{f} missing field 'text'")
        else:
            raise ValueError(f"Unsupported file type: {f}")
        ds_list.append(ds)

    return concatenate_datasets(ds_list) if len(ds_list) > 1 else ds_list[0]


def _group_texts(examples: Dict[str, list], block_size: int):
    keys = [k for k, v in examples.items() if isinstance(v, list) and v and isinstance(v[0], list)]
    concatenated = {k: sum(examples[k], []) for k in keys}
    total_len = len(concatenated.get("input_ids", []))
    total_len = (total_len // block_size) * block_size
    result = {k: [t[i:i + block_size] for i in range(0, total_len, block_size)] for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result


def _force_eval_key_compat(targs: Dict[str, Any]) -> None:
    if "eval_strategy" in targs and "evaluation_strategy" in targs:
        targs.pop("evaluation_strategy", None)

    params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    if "eval_strategy" in targs and "eval_strategy" not in params:
        v = targs.pop("eval_strategy")
        if "evaluation_strategy" in params:
            targs["evaluation_strategy"] = v
    elif "evaluation_strategy" in targs and "evaluation_strategy" not in params:
        v = targs.pop("evaluation_strategy")
        if "eval_strategy" in params:
            targs["eval_strategy"] = v


class ThroughputLogger(TrainerCallback):
    def __init__(self, tokens_per_step: int):
        self.tokens_per_step = tokens_per_step
        self.t0 = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.t0 is None:
            return
        dt = time.time() - self.t0
        tps = self.tokens_per_step / max(dt, 1e-6)
        if state.is_world_process_zero and (state.global_step % max(1, args.logging_steps) == 0):
            print(f"[Speed] step={state.global_step} ~{tps:,.0f} tokens/s")


class DynamicCheckpointSaver(TrainerCallback):
    def __init__(self, freq_list):
        self.freq_list = freq_list

    def _get_interval(self, step: int) -> Optional[int]:
        for interval, max_step in self.freq_list:
            if step <= max_step:
                return interval
        return None

    def on_step_end(self, args, state, control, **kwargs):
        interval = self._get_interval(state.global_step)
        if interval is not None and state.global_step % interval == 0:
            control.should_save = True
        return control


def _setup_env(cache_dir: Path):
    if "PYTORCH_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:64"

    hf_home = cache_dir
    hf_datasets = cache_dir / "hf_datasets"
    hf_hub = cache_dir / "hf_hub"
    tmpdir = cache_dir / "tmp"
    xdg = cache_dir / "xdg"

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets)
    os.environ["HF_HUB_CACHE"] = str(hf_hub)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_hub)
    os.environ["XDG_CACHE_HOME"] = str(xdg)
    os.environ["TMPDIR"] = str(tmpdir)
    os.environ["TMP"] = str(tmpdir)
    os.environ["TEMP"] = str(tmpdir)

    for p in [hf_home, hf_datasets, hf_hub, tmpdir, xdg]:
        p.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    run_id = cfg.get("run_id", cfg_path.stem)
    artifacts = cfg.get("artifacts", {})
    run_dir = Path(artifacts["run_dir"]).expanduser().resolve()
    cache_dir = Path(artifacts["cache_dir"]).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _setup_env(cache_dir)

    data_cfg: Dict[str, Any] = cfg.get("data", {})
    train_files: List[str] = data_cfg.get("train_files", [])
    eval_files: List[str] = data_cfg.get("eval_files", [])
    oversample_plan = data_cfg.get("oversample_plan", None)

    if oversample_plan:
        train_files = _load_and_oversample(train_files, oversample_plan)

    targs = dict(cfg.get("training_arguments", {}))
    model_name = cfg.get("model_name", "gpt2")
    block_size = int(cfg.get("block_size", 1024))
    seed = int(cfg.get("seed", 42))

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds = _build_dataset(train_files)
    if train_ds is None:
        raise ValueError("No train_files provided in config.")

    eval_ds = _build_dataset(eval_files) if eval_files else None

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=str(cache_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    to_add = [m for m in [AGENT_MARK, PATIENT_MARK] if m not in tokenizer.get_vocab()]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    gpt2_cfg = GPT2Config.from_pretrained(model_name, cache_dir=str(cache_dir))
    gpt2_cfg.vocab_size = len(tokenizer)

    if getattr(gpt2_cfg, "n_positions", None) is not None:
        gpt2_cfg.n_positions = max(int(gpt2_cfg.n_positions), block_size)
    if getattr(gpt2_cfg, "n_ctx", None) is not None:
        gpt2_cfg.n_ctx = max(int(gpt2_cfg.n_ctx), block_size)

    gpt2_cfg.pad_token_id = tokenizer.pad_token_id
    gpt2_cfg.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        gpt2_cfg.bos_token_id = tokenizer.bos_token_id

    model = GPT2LMHeadModel(gpt2_cfg)

    if bool(targs.get("gradient_checkpointing", False)):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False

    def tok_fn(batch):
        return tokenizer(batch["text"], return_attention_mask=False)

    cache_base = Path(cache_dir) / "datasets_cache"
    cache_base.mkdir(parents=True, exist_ok=True)

    train_tok = train_ds.map(
        tok_fn,
        batched=True,
        remove_columns=train_ds.column_names,
        num_proc=4,
        cache_file_name=str(cache_base / f"{run_id}_train_tok.arrow"),
    )
    train_tok = train_tok.map(
        lambda ex: _group_texts(ex, block_size),
        batched=True,
        batch_size=1000,
        num_proc=1,
        cache_file_name=str(cache_base / f"{run_id}_train_grouped.arrow"),
    )

    eval_tok = None
    if eval_ds is not None:
        eval_tok = eval_ds.map(
            tok_fn,
            batched=True,
            remove_columns=eval_ds.column_names,
            num_proc=4,
            cache_file_name=str(cache_base / f"{run_id}_eval_tok.arrow"),
        )
        eval_tok = eval_tok.map(
            lambda ex: _group_texts(ex, block_size),
            batched=True,
            batch_size=1000,
            num_proc=1,
            cache_file_name=str(cache_base / f"{run_id}_eval_grouped.arrow"),
        )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    _force_eval_key_compat(targs)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        seed=seed,
        disable_tqdm=False,
        **targs,
    )
    training_args.dataloader_persistent_workers = (
        getattr(training_args, "dataloader_num_workers", 0) > 0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    if "checkpoint_frequency" in cfg:
        trainer.add_callback(DynamicCheckpointSaver(cfg["checkpoint_frequency"]))

    tokens_per_step = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * block_size
    )
    trainer.add_callback(ThroughputLogger(tokens_per_step))

    resume = bool(cfg.get("resume", False))
    resume_checkpoint = cfg.get("resume_checkpoint", None)

    resume_from = None
    if resume:
        if not resume_checkpoint:
            raise ValueError("resume=true requires a non-null resume_checkpoint in YAML.")
        resume_from = str(Path(resume_checkpoint).expanduser().resolve())
        if not Path(resume_from).exists():
            raise FileNotFoundError(f"resume_checkpoint not found: {resume_from}")
    else:
        if any(run_dir.glob("checkpoint-*")):
            print(f"[Warn] Found existing checkpoints in {run_dir}, but resume=false. Starting from scratch anyway.")

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model()
    tokenizer.save_pretrained(run_dir)


if __name__ == "__main__":
    main()
