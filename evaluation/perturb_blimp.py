#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/perturb_blimp.py
Perturb one BLiMP jsonl dataset (sentence_good/sentence_bad) according to a run_id.

Usage
------
python -m evaluation.perturb_blimp --run_id localA_animacy_natural --blimp_in /path/in.jsonl --blimp_out /path/out.jsonl
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

import spacy
import benepar
from tqdm import tqdm

from utils import (
    MODEL_PATH,
    AGENT_MARK,
    PATIENT_MARK,
    SEMANTIC_HIERARCHIES,
    should_mark_local_A,
    should_mark_local_P,
    should_mark_global,
)

FEATURES = set(SEMANTIC_HIERARCHIES.keys())
LOW_VALUE = {k: v[0] for k, v in SEMANTIC_HIERARCHIES.items()}
HIGH_VALUE = {k: v[-1] for k, v in SEMANTIC_HIERARCHIES.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--blimp_in", required=True)
    ap.add_argument("--blimp_out", required=True)
    ap.add_argument("--model_path", type=str, default=MODEL_PATH)
    ap.add_argument("--spacy_model_size", choices=["sm", "trf"], default="sm")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--spacy_keep_tagger", action="store_true")
    ap.add_argument("--no_spacy_keep_tagger", action="store_false", dest="spacy_keep_tagger")
    ap.set_defaults(spacy_keep_tagger=True)

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_out", type=str, default=None)
    ap.add_argument("--debug_limit", type=int, default=None)
    ap.add_argument("--debug_only_interesting", action="store_true")
    return ap.parse_args()


def build_nlp(model_size: str = "sm", keep_tagger: bool = True):
    model_name = "en_core_web_trf" if model_size == "trf" else "en_core_web_sm"
    disable = ["ner", "lemmatizer"] if keep_tagger else ["ner", "tagger", "lemmatizer"]
    nlp = spacy.load(model_name, disable=disable)
    try:
        benepar.load_trained_model("benepar_en3")
    except Exception:
        benepar.download("benepar_en3")
    if "benepar" not in nlp.pipe_names:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"}, last=True)
    nlp.max_length = 5_000_000
    return nlp


def annotate_first_sentence(doc):
    sent = list(doc.sents)[0]
    sent_start = sent.start
    tokens = [{
        "id": int(tok.i - sent_start),
        "text": tok.text,
        "lemma": tok.lemma_,
        "pos": tok.pos_,
        "dep": tok.dep_,
        "head": int(tok.head.i - sent_start),
    } for tok in sent]
    ptb = getattr(sent._, "parse_string", "")
    return tokens, ptb


def build_children(tokens: List[dict]) -> Dict[int, List[dict]]:
    ch: Dict[int, List[dict]] = {}
    for t in tokens:
        hid = t.get("head", -1)
        if isinstance(hid, int):
            ch.setdefault(hid, []).append(t)
    return ch


def subtree_span(root_id: int, id2tok: Dict[int, dict], children: Dict[int, List[dict]]) -> Tuple[int, int]:
    stack = [root_id]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for c in children.get(cur, []):
            cid = c.get("id")
            if isinstance(cid, int) and cid not in seen:
                stack.append(cid)
    ids = [i for i in seen if i in id2tok]
    if not ids:
        return root_id, root_id
    return min(ids), max(ids)


def extract_np(token_id: int, id2tok: Dict[int, dict], children: Dict[int, List[dict]]) -> dict:
    s, e = subtree_span(token_id, id2tok, children)
    head = id2tok[token_id]
    return {"span": [s, e], "head": head.get("text", ""), "head_pos": head.get("pos")}


def extract_clause(token_id: int, id2tok: Dict[int, dict], children: Dict[int, List[dict]]) -> dict:
    s, e = subtree_span(token_id, id2tok, children)
    head = id2tok[token_id]
    return {"span": [s, e], "head": head.get("text", ""), "head_pos": head.get("pos")}


def extract_verb_args(tokens: List[dict]) -> List[dict]:
    if not tokens:
        return []
    id2tok = {t["id"]: t for t in tokens if isinstance(t.get("id"), int)}
    children = build_children(tokens)
    verbs = [t for t in tokens if t.get("pos") in {"VERB", "AUX"}]
    out = []
    for v in verbs:
        vid = v["id"]
        rec = {"verb": v.get("lemma", v.get("text", "")), "verb_id": vid, "subject": None, "objects": []}
        for ch in children.get(vid, []):
            dep = ch.get("dep")
            tid = ch.get("id")
            if not isinstance(tid, int):
                continue
            if dep in {"nsubj", "nsubjpass", "csubj"}:
                rec["subject"] = extract_np(tid, id2tok, children)
            elif dep in {"dobj", "obj", "attr", "dative"}:
                sp = extract_np(tid, id2tok, children)
                sp["dep"] = dep
                rec["objects"].append(sp)
            elif dep in {"xcomp", "ccomp"}:
                sp = extract_clause(tid, id2tok, children)
                sp["dep"] = dep
                rec["objects"].append(sp)
        out.append(rec)
    return out


def is_valid_structure(entry: dict) -> bool:
    return bool(entry.get("subject")) and len(entry.get("objects", [])) == 1 and entry["objects"][0].get("dep") != "ccomp"


def detok(tokens: List[dict]) -> str:
    return " ".join(t["text"] for t in tokens)


def detok_span(tokens: List[dict], span: List[int]) -> str:
    s, e = span
    return " ".join(tokens[i]["text"] for i in range(s, e + 1) if 0 <= i < len(tokens))


def apply_spans(tokens: List[dict], spans: List[dict]) -> str:
    if not spans:
        return detok(tokens)
    spans = sorted(spans, key=lambda x: x["span"][0])
    out, i, n, last_end = [], 0, len(tokens), -1
    for sp in spans:
        s, e = sp["span"]
        if not (0 <= s <= e < n):
            continue
        if s <= last_end:
            continue
        while i < s:
            out.append(tokens[i]["text"])
            i += 1
        out.append(sp["text"])
        i = e + 1
        last_end = e
    while i < n:
        out.append(tokens[i]["text"])
        i += 1
    return " ".join(out)


def strip_marks(s: str) -> str:
    return " ".join(s.replace(AGENT_MARK, "").replace(PATIENT_MARK, "").split()).strip()


def parse_run_id(run_id: str) -> Tuple[str, Optional[str], Optional[str]]:
    rule = run_id.split("_")[0]
    if rule in {"baseline", "full"}:
        return rule, None, None
    parts = run_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"run_id must be <rule>_<feature>_<direction>, got: {run_id}")
    rule, feature, direction = parts[0], parts[1], parts[2]
    if rule not in {"localA", "localP", "global"}:
        raise ValueError(f"bad rule: {run_id}")
    if feature not in FEATURES:
        raise ValueError(f"bad feature: {run_id}")
    if direction not in {"natural", "inverse"}:
        raise ValueError(f"bad direction: {run_id}")
    return rule, feature, direction


def decide_marks(run_id: str, A: Dict[str, Optional[str]], P: Dict[str, Optional[str]], dbg: Optional[dict] = None) -> Tuple[bool, bool]:
    rule, feature, direction = parse_run_id(run_id)
    if rule == "baseline":
        if dbg is not None:
            dbg.update({"type": "baseline"})
        return False, False
    if rule == "full":
        if dbg is not None:
            dbg.update({"type": "full"})
        return True, True
    if rule == "localA":
        mA = should_mark_local_A(A, feature, direction)
        if dbg is not None:
            dbg.update({"type": "localA", "feature": feature, "direction": direction, "A": A.get(feature), "markA": mA})
        return bool(mA), False
    if rule == "localP":
        mP = should_mark_local_P(P, feature, direction)
        if dbg is not None:
            dbg.update({"type": "localP", "feature": feature, "direction": direction, "P": P.get(feature), "markP": mP})
        return False, bool(mP)
    mA, mP = should_mark_global(A, P, feature, direction)
    if dbg is not None:
        dbg.update({"type": "global", "feature": feature, "direction": direction, "A": A.get(feature), "P": P.get(feature), "mark": (mA and mP)})
    return bool(mA), bool(mP)


class NPLabelClassifier:
    def __init__(self, model_dir: Path, id2label: Dict[int, str], device: str, max_length: int = 128):
        self.tokenizer = BertTokenizer.from_pretrained(str(model_dir))
        self.model = BertForSequenceClassification.from_pretrained(str(model_dir)).to(device)
        self.model.eval()
        self.id2label = dict(id2label)
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def predict(self, sentence: str, np_text: str) -> Optional[str]:
        inp = f"{sentence} [NP] {np_text}"
        enc = self.tokenizer(
            inp, add_special_tokens=True, max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt",
        ).to(self.device)
        logits = self.model(enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        pred = int(torch.argmax(logits, dim=1).item())
        return self.id2label.get(pred)


class NPLabeler:
    def __init__(self, model_path: Path, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        base = Path(model_path)

        self.anim = NPLabelClassifier(base / "animacy_bert_model", {0: "animate", 1: "inanimate"}, self.device)
        self.defi = NPLabelClassifier(base / "definiteness_bert_model", {0: "definite", 1: "indef"}, self.device)
        self.pron = NPLabelClassifier(base / "pronominality_bert_model", {0: "pronoun", 1: "common"}, self.device)

    def label(self, sent: str, np_text: str) -> Dict[str, Optional[str]]:
        return {
            "animacy": self.anim.predict(sent, np_text),
            "definiteness": self.defi.predict(sent, np_text),
            "pronominality": self.pron.predict(sent, np_text),
        }


def perturb_one(sent_text: str, nlp, lab: NPLabeler, run_id: str) -> Tuple[str, dict]:
    tr = {"original": sent_text, "run_id": run_id, "reason": None, "predicates": []}
    if not sent_text.strip():
        tr["reason"] = "empty"
        return sent_text, tr

    doc = nlp(sent_text)
    tokens, ptb = annotate_first_sentence(doc)
    base = detok(tokens)
    verbs = extract_verb_args(tokens)
    if not verbs:
        tr["reason"] = "no_verbs"
        return sent_text, tr

    spans = []
    for ent in verbs:
        et = {"verb": ent.get("verb"), "rule_debug": {}}
        if not is_valid_structure(ent):
            et["note"] = "invalid_structure"
            tr["predicates"].append(et)
            continue

        subj = ent["subject"]
        obj = ent["objects"][0]
        sA, eA = subj["span"]
        sP, eP = obj["span"]

        A_np = detok_span(tokens, [sA, eA])
        P_np = detok_span(tokens, [sP, eP])

        A = lab.label(base, A_np)
        P = lab.label(base, P_np)

        mA, mP = decide_marks(run_id, A, P, dbg=et["rule_debug"])
        et.update({"A": A, "P": P, "A_span": [sA, eA], "P_span": [sP, eP], "markA": mA, "markP": mP})
        tr["predicates"].append(et)

        if mA:
            spans.append({"span": [sA, eA], "text": f"{A_np} {AGENT_MARK}"})
        if mP:
            spans.append({"span": [sP, eP], "text": f"{P_np} {PATIENT_MARK}"})

    if not spans:
        tr["reason"] = "no_trigger"
        return sent_text, tr

    new_sent = apply_spans(tokens, spans)
    if strip_marks(new_sent) != strip_marks(base):
        tr["reason"] = "minimality_failed"
        tr["new_detok"] = new_sent
        return sent_text, tr

    tr["reason"] = "marked"
    tr["new_detok"] = new_sent
    return new_sent, tr


def main():
    args = parse_args()
    blimp_in = Path(args.blimp_in).resolve()
    blimp_out = Path(args.blimp_out).resolve()
    blimp_out.parent.mkdir(parents=True, exist_ok=True)

    nlp = build_nlp(args.spacy_model_size, args.spacy_keep_tagger)
    lab = NPLabeler(Path(args.model_path), device=args.device)

    fout_dbg = None
    dbg_path = None
    if args.debug:
        dbg_path = Path(args.debug_out) if args.debug_out else blimp_out.with_suffix(".debug.jsonl")
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        fout_dbg = dbg_path.open("w", encoding="utf-8")

    def interesting(tr: dict) -> bool:
        if tr.get("reason") == "marked":
            return True
        for pr in tr.get("predicates", []):
            if pr.get("note"):
                return True
            if pr.get("markA") or pr.get("markP"):
                return True
        return False

    n_pairs = 0
    n_dbg = 0
    stop = False

    with blimp_in.open("r", encoding="utf-8") as fin, blimp_out.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"Perturbing {blimp_in.name} [{args.run_id}]"):
            if stop:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            g0 = data.get("sentence_good", "")
            b0 = data.get("sentence_bad", "")

            g1, tg = perturb_one(g0, nlp, lab, args.run_id)
            b1, tb = perturb_one(b0, nlp, lab, args.run_id)

            out_obj = {"sentence_good": g1, "sentence_bad": b1}
            for k in ["UID", "pairID", "field", "linguistics_term"]:
                if k in data:
                    out_obj[k] = data[k]
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n_pairs += 1

            if fout_dbg is not None:
                for rec in ({"type": "good", **tg}, {"type": "bad", **tb}):
                    if args.debug_only_interesting and not interesting(rec):
                        continue
                    fout_dbg.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_dbg += 1
                    if args.debug_limit is not None and n_dbg >= args.debug_limit:
                        stop = True
                        break

    if fout_dbg is not None:
        fout_dbg.close()
        print(f"[DEBUG] wrote traces: {n_dbg} -> {dbg_path}")
    print(f"[DONE] pairs={n_pairs} -> {blimp_out}")


if __name__ == "__main__":
    main()
