#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifiers/train_classifier.py
Train BERT classifiers for:
  - animacy (animate / inanimate)
  - definiteness (definite / indef)
  - pronominality (pronoun / common)

Usage
------
python -m classifiers.train_classifier --task animacy
"""

import os
import argparse
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification

from utils import DATA_PATH, MODEL_PATH


TASK_CONFIGS = {
    "animacy": {
        "field": "animacy",
        "labels": {"animate": 0, "inanimate": 1},
        "targets": ["animate", "inanimate"],
        "save_dir": "animacy_bert_model",
    },
    "pronominality": {
        "field": "pronominality",
        "labels": {"pronoun": 0, "common": 1},
        "targets": ["pronoun", "common"],
        "save_dir": "pronominality_bert_model",
    },
    "definiteness": {
        "field": "definiteness",
        "labels": {"definite": 0, "indef": 1},
        "targets": ["definite", "indef"],
        "save_dir": "definiteness_bert_model",
    },
}


def parse_args():
    ap = argparse.ArgumentParser(description="Train BERT classifiers for NP-level semantic features.")
    ap.add_argument("--task", choices=["animacy", "pronominality", "definiteness"], required=True,
                    help="Semantic classification task.")
    ap.add_argument("--csv", type=str, default=None,
                    help="Path to training CSV. Default: auto-pick training_data_*<task>*.csv under DATA_PATH.")
    ap.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    ap.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    ap.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    ap.add_argument("--max-length", type=int, default=128, help="Maximum input length for BERT.")
    ap.add_argument("--test-size", type=float, default=0.2, help="Proportion of data used for held-out evaluation.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    return ap.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def auto_pick_csv(task: str) -> str:
    """Automatically find training_data_*<task>*.csv under DATA_PATH."""
    for fname in os.listdir(DATA_PATH):
        if fname.endswith(".csv") and task in fname:
            return os.path.join(DATA_PATH, fname)
    raise FileNotFoundError(
        f"No training_data_*{task}*.csv found in {DATA_PATH}. "
        f"Please generate training data first."
    )


class NPTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_length: int, field: str):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.field = field

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"{row['sentence']} [NP] {row['np']}"
        label = int(row[self.field])
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = TASK_CONFIGS[args.task]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pick CSV
    csv_path = args.csv or auto_pick_csv(args.task)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    print(f"[INFO] Using CSV: {csv_path}")

    # Load and clean data
    df = pd.read_csv(csv_path)
    needed = {"sentence", "np", "np_role", cfg["field"]}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df[df[cfg["field"]].isin(cfg["labels"].keys())].copy()
    df.dropna(subset=["sentence", "np", cfg["field"]], inplace=True)
    df.drop_duplicates(subset=["sentence", "np", "np_role", cfg["field"]], inplace=True)
    df[cfg["field"]] = df[cfg["field"]].map(cfg["labels"])

    if len(df) < 100:
        print(f"[WARN] Very small dataset (n={len(df)}). Consider generating more examples.")

    # Label distribution
    dist = Counter(df[cfg["field"]].tolist())
    inv_label = {v: k for k, v in cfg["labels"].items()}
    pretty = {inv_label[k]: v for k, v in dist.items()}
    print(f"[INFO] Label distribution: {pretty}")

    # Stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df[cfg["field"]] if len(dist) > 1 else None,
    )
    print(f"[INFO] Train size={len(train_df)}, Test size={len(test_df)}")

    # Tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = NPTaskDataset(train_df, tokenizer, args.max_length, cfg["field"])
    test_ds = NPTaskDataset(test_df, tokenizer, args.max_length, cfg["field"])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(cfg["labels"]),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[INFO] Epoch {epoch} average loss: {avg_loss:.4f}")

    # Save model
    save_dir = os.path.join(MODEL_PATH, cfg["save_dir"])
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)
    print(f"[INFO] Model saved to: {save_dir}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=cfg["targets"],
            digits=4,
        )
    )


if __name__ == "__main__":
    main()
