#!/usr/bin/env python3
"""
Step 4 — Counterfactual analysis.

Reads ONLY predictions_counterfactual_{model}.jsonl (from Step 2) and
counterfactual_eval.jsonl. No inference happens here.

For each transform (order_permutation, timestamp_collapse,
identity_substitution) and each model: N, mean/median original
probability, mean/median counterfactual probability, mean/median change,
fraction moving in the expected direction. "Expected direction" = a
decrease, since every base sample here is a genuine positive
(n_occ_intended >= 2) and every transform is designed to break either
order, persistence, or cross-site identity consistency.

Usage:
    python step4_counterfactual_analysis.py --data-dir <data_dir> --out-dir <out_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common

MODELS = ["baseline", "v3a", "v3c"]


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

    cf_rows = common.load_counterfactual(args.data_dir / "counterfactual_eval.jsonl")
    cf_meta = {r["sample_id"]: r for r in cf_rows}

    all_rows = []
    per_model_pairs = {}

    for model in MODELS:
        preds = {r["sample_id"]: r["prob"] for r in common.load_jsonl(out / f"predictions_counterfactual_{model}.jsonl")}

        base_probs = {sid: preds[sid] for sid, r in cf_meta.items() if r["role"] == "base"}

        pairs = []
        for sid, r in cf_meta.items():
            if r["role"] != "transformed":
                continue
            base_sid = r["base_sample_id"]
            if base_sid not in base_probs:
                continue
            pairs.append({
                "transform": r["transform"],
                "base_sample_id": base_sid,
                "counterfactual_sample_id": sid,
                "original_prob": base_probs[base_sid],
                "counterfactual_prob": preds[sid],
                "change": preds[sid] - base_probs[base_sid],
            })
        pdf = pd.DataFrame(pairs)
        per_model_pairs[model] = pdf

        for transform, sub in pdf.groupby("transform"):
            moved_expected = float((sub.change < 0).mean())  # expected direction = decrease
            row = {
                "model": model,
                "transform": transform,
                "n": int(len(sub)),
                "mean_original_prob": float(sub.original_prob.mean()),
                "median_original_prob": float(sub.original_prob.median()),
                "mean_counterfactual_prob": float(sub.counterfactual_prob.mean()),
                "median_counterfactual_prob": float(sub.counterfactual_prob.median()),
                "mean_change": float(sub.change.mean()),
                "median_change": float(sub.change.median()),
                "fraction_moving_expected_direction_decrease": moved_expected,
            }
            all_rows.append(row)

    out_df = pd.DataFrame(all_rows).sort_values(["transform", "model"])
    out_df.to_csv(out / "counterfactual_analysis.csv", index=False)
    common.write_json(out / "counterfactual_analysis.json", all_rows)

    # also save the raw per-pair table for v3a/v3c (useful for step 7 / manual inspection)
    for model in ["v3a", "v3c"]:
        per_model_pairs[model].to_csv(out / f"counterfactual_pairs_{model}.csv", index=False)

    print(out_df.to_string(index=False))
    print(f"\nWrote: {out / 'counterfactual_analysis.json'}")
    print(f"Wrote: {out / 'counterfactual_analysis.csv'}")


if __name__ == "__main__":
    main()
