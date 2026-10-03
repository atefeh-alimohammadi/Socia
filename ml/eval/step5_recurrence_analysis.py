#!/usr/bin/env python3
"""
Step 5 — Recurrence analysis.

For POSITIVE test examples only, analyze predicted probability and
recall as a function of n_occ_intended (2, 3, 4). Reads only saved
predictions_test_{model}.jsonl + test_eval_metadata.jsonl. No inference.

Reports a simple descriptive measure of whether probability rises with
n_occ_intended (mean per bucket + Spearman rho over the three bucket
means) WITHOUT any significance claim, per the brief.

Usage:
    python step5_recurrence_analysis.py --data-dir <data_dir> --out-dir <out_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common

MODELS = ["baseline_timing_category", "v3a", "v3c"]
FILE_NAME_FOR_MODEL = {"baseline_timing_category": "baseline", "v3a": "v3a", "v3c": "v3c"}


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
        final_eval = json.load(f)

    all_rows = []
    for model_key in MODELS:
        fname = FILE_NAME_FOR_MODEL[model_key]
        preds = common.load_jsonl(out / f"predictions_test_{fname}.jsonl")
        threshold = final_eval["models"][model_key]["threshold"]

        df = pd.DataFrame(preds)
        df = df[df.label == 1].copy()
        df["n_occ_intended"] = df["sample_id"].map(lambda sid: meta[sid]["n_occ_intended"])
        df["pred_positive"] = (df["prob"] >= threshold).astype(int)

        bucket_means = []
        for bucket, sub in sorted(df.groupby("n_occ_intended")):
            n = len(sub)
            row = {
                "model": model_key,
                "n_occ_intended": int(bucket),
                "n": n,
                "mean_pred_prob": float(sub.prob.mean()),
                "median_pred_prob": float(sub.prob.median()),
                "recall_at_threshold": float(sub.pred_positive.mean()),
                "threshold_used": float(threshold),
            }
            all_rows.append(row)
            bucket_means.append((bucket, row["mean_pred_prob"]))

        # descriptive-only monotonicity measure
        buckets_sorted = sorted(bucket_means, key=lambda x: x[0])
        means_in_order = [m for _, m in buckets_sorted]
        monotonic_nondecreasing = all(means_in_order[i] <= means_in_order[i + 1] + 1e-9
                                       for i in range(len(means_in_order) - 1))
        if len(buckets_sorted) >= 2:
            rho, _ = spearmanr([b for b, _ in buckets_sorted], means_in_order)
        else:
            rho = None
        for row in all_rows:
            if row["model"] == model_key:
                row["mean_prob_monotonic_nondecreasing_across_buckets"] = monotonic_nondecreasing
                row["spearman_rho_bucket_vs_mean_prob_descriptive_only"] = None if rho is None else float(rho)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out / "recurrence_analysis.csv", index=False)
    common.write_json(out / "recurrence_analysis.json", all_rows)

    print(out_df.to_string(index=False))
    print("\nNote: monotonicity/Spearman rho are DESCRIPTIVE ONLY — no significance claim.")
    print(f"\nWrote: {out / 'recurrence_analysis.json'}")
    print(f"Wrote: {out / 'recurrence_analysis.csv'}")


if __name__ == "__main__":
    main()
