"""
Socia ML final evaluation — shared utilities.

Nothing in this file trains, retrains, or modifies any model, dataset, or
checkpoint. It only loads frozen artifacts and runs inference/metrics.

This module needs `torch` and the frozen project code (model.py,
model_v2.py, model_v3a.py, model_v3c.py, dataset_v2.py) importable. Point
--code-dir at the directory containing them (see step2_final_evaluation.py
--help). It does NOT need `dataset.py`'s strict validators for the
counterfactual file (see load_counterfactual below) because that file
intentionally has a different, evaluation-only schema.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Path / import setup
# ---------------------------------------------------------------------------

def add_code_dirs_to_path(*dirs: Path) -> None:
    for d in dirs:
        d = str(Path(d).resolve())
        if d not in sys.path:
            sys.path.insert(0, d)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_eval_metadata_by_id(path: Path) -> Dict[str, dict]:
    return {m["sample_id"]: m for m in load_jsonl(path)}


def load_counterfactual(path: Path) -> List[dict]:
    """
    counterfactual_eval.jsonl has a DIFFERENT schema from train/val/test:
    role="base" rows carry "label"; role="transformed" rows carry
    "expected_label" instead (the label a correctly-behaving model SHOULD
    move toward, not a ground-truth training label). dataset.py's
    validate_record() would reject transformed rows outright (it requires
    a "label" key). That is a genuine, real schema difference, not a bug
    to patch around silently — this loader handles it explicitly instead
    of running these rows through dataset.py's validator.

    Also does its own minimal structural check (25 events, valid
    category_id range, delta_hours finite/non-negative/non-decreasing)
    so a malformed row fails loudly here rather than silently in the
    model, but does NOT require the "label" key to be present.
    """
    rows = load_jsonl(path)
    for i, r in enumerate(rows):
        events = r.get("events")
        if not isinstance(events, list) or len(events) != 25:
            raise ValueError(f"counterfactual row {i} ({r.get('sample_id')}): expected 25 events.")
        prev = None
        for j, e in enumerate(events):
            if not (0 <= e["category_id"] < 15):
                raise ValueError(f"counterfactual row {i} event {j}: category_id out of range.")
            dh = float(e["delta_hours"])
            if not math.isfinite(dh) or dh < 0:
                raise ValueError(f"counterfactual row {i} event {j}: delta_hours invalid ({dh}).")
            if prev is not None and dh < prev:
                raise ValueError(f"counterfactual row {i} event {j}: delta_hours not non-decreasing.")
            prev = dh
        if r["role"] == "base" and "label" not in r:
            raise ValueError(f"counterfactual row {i}: base row missing 'label'.")
        if r["role"] == "transformed" and "expected_label" not in r:
            raise ValueError(f"counterfactual row {i}: transformed row missing 'expected_label'.")
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Model loading + inference  (imports torch / project code lazily, so this
# module can still be imported by pure-analysis steps that don't need torch)
# ---------------------------------------------------------------------------

MODEL_VERSIONS = ("v3a", "v3c")


def load_model(version: str, checkpoint_path: Path, device=None):
    """
    Returns (model, checkpoint_meta_dict). `checkpoint_meta_dict` excludes
    the raw state_dict (too large / not useful downstream) but keeps
    epoch, val_f1, val_threshold, val_precision, val_recall, time_stats,
    seed, model_version exactly as stored at training time.
    """
    import torch

    if version == "v3a":
        from model_v3a import SociaPatternTransformerV3A as ModelClass
    elif version == "v3c":
        from model_v3c import SociaPatternTransformerV3C as ModelClass
    else:
        raise ValueError(f"Unknown model version: {version}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ModelClass().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, meta, device


def records_to_tensors(records: List[dict], time_stats: Dict[str, float], device):
    """
    Builds (category_ids, temporal_features, raw_delta_hours) tensors for
    a list of raw dataset records, using dataset_v2's OWN feature-
    engineering function unchanged (imported, not reimplemented) so eval
    features are guaranteed identical to training features.
    """
    import torch
    from dataset_v2 import compute_temporal_features_single

    cat_batch, temp_batch, raw_batch = [], [], []
    for r in records:
        events = r["events"]
        cat_ids = [int(e["category_id"]) for e in events]
        raw_hours = [float(e["delta_hours"]) for e in events]
        temp = compute_temporal_features_single(cat_ids, raw_hours, time_stats)
        cat_batch.append(torch.tensor(cat_ids, dtype=torch.long))
        temp_batch.append(temp)
        raw_batch.append(torch.tensor(raw_hours, dtype=torch.float32))

    category_ids = torch.stack(cat_batch).to(device)
    temporal_features = torch.stack(temp_batch).to(device)
    raw_delta_hours = torch.stack(raw_batch).to(device)
    return category_ids, temporal_features, raw_delta_hours


def run_inference(model, records: List[dict], time_stats: Dict[str, float], device,
                   batch_size: int = 128, id_key: str = "sample_id") -> Dict[str, float]:
    """Returns {sample_id: predicted_probability}. Order of `records` does
    not matter; results are keyed by id, not position."""
    import torch

    probs: Dict[str, float] = {}
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            category_ids, temporal_features, raw_delta_hours = records_to_tensors(batch, time_stats, device)
            logits = model(category_ids, temporal_features, raw_delta_hours, check_finite=True)
            batch_probs = torch.sigmoid(logits).cpu().tolist()
            for r, p in zip(batch, batch_probs):
                probs[r[id_key]] = float(p)
    return probs


# ---------------------------------------------------------------------------
# Thresholding — mirrors train_v3a.py / train_v3c.py's own sweep exactly,
# so a val-selected threshold recomputed here is reproducible against what
# training already selected, not a new/different protocol.
# ---------------------------------------------------------------------------

def get_thresholds(lo: float = 0.10, hi: float = 0.90, step: float = 0.05) -> List[float]:
    out = []
    cur = lo
    while cur <= hi + 1e-9:
        out.append(round(cur, 2))
        cur += step
    return out


def sweep_best_threshold(labels: List[int], probs: List[float], thresholds: List[float]) -> Tuple[float, float, float, float]:
    """Returns (best_threshold, precision, recall, f1) selecting by F1,
    exactly mirroring run_validation() in train_v3a.py / train_v3c.py."""
    from sklearn.metrics import precision_recall_fscore_support

    best_f1, best_t, best_p, best_r = -1.0, 0.5, 0.0, 0.0
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in probs]
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = float(f1), t, float(p), float(r)
    return best_t, best_p, best_r, best_f1


def compute_full_metrics(labels: List[int], probs: List[float], threshold: float) -> dict:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix,
    )
    labels_arr = np.array(labels)
    probs_arr = np.array(probs)
    preds = (probs_arr >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(labels_arr, preds, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels_arr, preds, labels=[0, 1]).ravel()

    return {
        "n": int(len(labels_arr)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels_arr, probs_arr)),
        "pr_auc": float(average_precision_score(labels_arr, probs_arr)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def align_predictions(records: List[dict], probs: Dict[str, float], id_key: str = "sample_id",
                       label_key: str = "label") -> Tuple[List[int], List[float], List[str]]:
    """Returns (labels, probs, sample_ids) in a fixed, shared order derived
    from `records` — use the SAME records list for every model being
    compared so labels/order line up identically across baseline/v3a/v3c."""
    labels, plist, ids = [], [], []
    for r in records:
        sid = r[id_key]
        if sid not in probs:
            raise KeyError(f"No prediction for {sid}")
        labels.append(int(r[label_key]))
        plist.append(probs[sid])
        ids.append(sid)
    return labels, plist, ids
