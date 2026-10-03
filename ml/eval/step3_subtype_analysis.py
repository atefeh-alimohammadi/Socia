#!/usr/bin/env python3
"""
Step 3 — Subtype analysis.

Reads ONLY the predictions_test_{model}.jsonl files written by Step 2 and
test_eval_metadata.jsonl. No model inference happens here.

For each model x each of the six negative subtypes (plus a "positive"
row for context): N, false-positive count, false-positive rate, mean
predicted probability, median predicted probability, percent classified
positive at the validation-selected threshold.

Usage:
    python step3_subtype_analysis.py --data-dir <data_dir> --out-dir <out_dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common

MODELS = ["baseline_timing_category", "v3a", "v3c"]
FILE_NAME_FOR_MODEL = {"baseline_timing_category": "baseline", "v3a": "v3a", "v3c": "v3c"}

SUBTYPES = [
    "positive",
    "timing_matched",
    "order_permutation",
    "boundary_single_occurrence",
    "category_identity",
    "boundary_tight_burst",
    "pure_background",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\AI Companion\Socia\ml\dataset\scripts\data"),
        help="Directory containing test_eval_metadata.jsonl",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"D:\AI Companion\Socia\ml\eval\eval_outputs"),
        help="Directory containing Step 2 prediction outputs and where Step 3 outputs will be written",
    )
    return p.parse_args()

def main():
    args = parse_args()
    out = args.out_dir

    meta = common.load_eval_metadata_by_id(args.data_dir / "test_eval_metadata.jsonl")

    with (out / "final_evaluation_report.json").open("r", encoding="utf-8") as f:
        import json
        final_eval = json.load(f)

    all_rows = []
    for model_key in MODELS:
        fname = FILE_NAME_FOR_MODEL[model_key]
        preds = common.load_jsonl(out / f"predictions_test_{fname}.jsonl")
        threshold = final_eval["models"][model_key]["threshold"]

        df = pd.DataFrame(preds)
        df["subtype"] = df["sample_id"].map(lambda sid: meta[sid]["subtype"])
        df["pred_positive"] = (df["prob"] >= threshold).astype(int)

        for subtype in SUBTYPES:
            sub = df[df.subtype == subtype]
            if len(sub) == 0:
                continue
            is_negative_subtype = subtype != "positive"
            n = len(sub)
            pct_pred_positive = float(sub.pred_positive.mean())
            row = {
                "model": model_key,
                "subtype": subtype,
                "n": n,
                "mean_pred_prob": float(sub.prob.mean()),
                "median_pred_prob": float(sub.prob.median()),
                "pct_classified_positive_at_threshold": pct_pred_positive,
                "threshold_used": float(threshold),
            }
            if is_negative_subtype:
                fp_count = int(sub.pred_positive.sum())
                row["false_positive_count"] = fp_count
                row["false_positive_rate"] = float(fp_count / n)
            else:
                # for the positive subtype, the analogous number is recall
                tp_count = int(sub.pred_positive.sum())
                row["true_positive_count"] = tp_count
                row["recall_within_positive"] = float(tp_count / n)
            all_rows.append(row)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out / "subtype_analysis.csv", index=False)
    common.write_json(out / "subtype_analysis.json", all_rows)

    print(out_df.to_string(index=False))
    print(f"\nWrote: {out / 'subtype_analysis.json'}")
    print(f"Wrote: {out / 'subtype_analysis.csv'}")


if __name__ == "__main__":
    main()
