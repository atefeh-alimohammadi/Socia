"""
The timing+category shortcut baseline, EXACT same feature set and model
(HistGradientBoostingClassifier) as diagnostics.py used earlier in this
project (the numbers this reproduces should match diagnostics_report.json's
"timing+category" entry: ROC-AUC 0.8094, PR-AUC 0.7632, F1@val_thr 0.7615,
thr 0.35 — this file being re-run here is itself the "under the same
protocol" check requested for Step 2).

No torch dependency — this baseline only needs numpy/pandas/sklearn, so it
is one part of the pipeline that CAN be run and verified without a torch
install.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CATEGORIES_N = 15


def featurize(record: dict) -> dict:
    ev = record["events"]
    delta = np.array([e["delta_hours"] for e in ev], dtype=float)
    cats = [e["category_id"] for e in ev]
    gaps = np.diff(delta)

    f = {"duration_h": delta[-1]}
    f["gap_mean"] = gaps.mean(); f["gap_median"] = np.median(gaps); f["gap_std"] = gaps.std()
    f["gap_min"] = gaps.min(); f["gap_max"] = gaps.max()
    for th in (0.5, 1, 3, 6, 12, 24):
        f[f"n_gap_lt_{th}h"] = int((gaps < th).sum())
    cnt = np.bincount(cats, minlength=CATEGORIES_N)
    for i, c in enumerate(cnt):
        f[f"cat_{i:02d}"] = int(c)
    f["n_unique_cat"] = int((cnt > 0).sum())
    f["max_cat_count"] = int(cnt.max())
    p = cnt[cnt > 0] / cnt.sum()
    f["cat_entropy"] = float(-(p * np.log(p)).sum())
    return f


def _cols(df: pd.DataFrame) -> List[str]:
    timing = [c for c in df.columns if c.startswith(("duration", "gap_", "n_gap_lt"))]
    category = [c for c in df.columns if c.startswith(("cat_", "n_unique", "max_cat"))]
    return timing + category


def build_feature_frame(records: List[dict], id_key: str = "sample_id",
                         label_key: str = "label") -> pd.DataFrame:
    rows = []
    for r in records:
        f = featurize(r)
        f[id_key] = r[id_key]
        if label_key in r:
            f["label"] = int(r[label_key])
        rows.append(f)
    return pd.DataFrame(rows)


def fit_baseline(train_records: List[dict]):
    df = build_feature_frame(train_records)
    cols = _cols(df)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    model.fit(df[cols], df["label"])
    return model, cols


def predict_baseline(model, cols: List[str], records: List[dict],
                      id_key: str = "sample_id") -> Dict[str, float]:
    df = build_feature_frame(records, id_key=id_key)
    probs = model.predict_proba(df[cols])[:, 1]
    return {sid: float(p) for sid, p in zip(df[id_key], probs)}
