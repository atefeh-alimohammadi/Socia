#!/usr/bin/env python3
"""
Step 6 — Error analysis.

For each model, breaks down false positives by negative subtype and
false negatives by recurrence-count bucket (n_occ_intended), reading
only predictions_test_{model}.jsonl + test_eval_metadata.jsonl. No
inference happens here.

Usage:
    python step6_error_analysis.py --data-dir <data_dir> --out-dir <out_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

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

    summary = {}
    for model_key in MODELS:
        fname = FILE_NAME_FOR_MODEL[model_key]
        preds = common.load_jsonl(out / f"predictions_test_{fname}.jsonl")
        threshold = final_eval["models"][model_key]["threshold"]

        df = pd.DataFrame(preds)
        df["subtype"] = df["sample_id"].map(lambda sid: meta[sid]["subtype"])
        df["n_occ_intended"] = df["sample_id"].map(lambda sid: meta[sid].get("n_occ_intended"))
        df["pred_positive"] = (df["prob"] >= threshold).astype(int)

        fp = df[(df.label == 0) & (df.pred_positive == 1)]
        fn = df[(df.label == 1) & (df.pred_positive == 0)]

        fp_by_subtype = fp.subtype.value_counts().to_dict()
        fp_by_subtype_share = (fp.subtype.value_counts(normalize=True).round(4)).to_dict() if len(fp) else {}
        fn_by_bucket = fn.n_occ_intended.value_counts().sort_index().to_dict()
        fn_by_bucket_share = (fn.n_occ_intended.value_counts(normalize=True).round(4).sort_index()).to_dict() if len(fn) else {}

        summary[model_key] = {
            "threshold_used": float(threshold),
            "total_n": int(len(df)),
            "total_fp": int(len(fp)),
            "total_fn": int(len(fn)),
            "fp_rate_overall": float(len(fp) / max((df.label == 0).sum(), 1)),
            "fn_rate_overall": float(len(fn) / max((df.label == 1).sum(), 1)),
            "fp_count_by_negative_subtype": {str(k): int(v) for k, v in fp_by_subtype.items()},
            "fp_share_of_all_fps_by_negative_subtype": {str(k): float(v) for k, v in fp_by_subtype_share.items()},
            "fn_count_by_n_occ_intended": {str(k): int(v) for k, v in fn_by_bucket.items()},
            "fn_share_of_all_fns_by_n_occ_intended": {str(k): float(v) for k, v in fn_by_bucket_share.items()},
        }

        print(f"\n=== {model_key} (threshold={threshold}) ===")
        print(f"Total FP: {len(fp)} / {int((df.label==0).sum())} negatives  "
              f"({summary[model_key]['fp_rate_overall']:.3f})")
        print(f"Total FN: {len(fn)} / {int((df.label==1).sum())} positives  "
              f"({summary[model_key]['fn_rate_overall']:.3f})")
        print("FP share by negative subtype:", summary[model_key]["fp_share_of_all_fps_by_negative_subtype"])
        print("FN share by n_occ_intended:  ", summary[model_key]["fn_share_of_all_fns_by_n_occ_intended"])

    common.write_json(out / "error_analysis.json", summary)
    print(f"\nWrote: {out / 'error_analysis.json'}")


if __name__ == "__main__":
    main()
