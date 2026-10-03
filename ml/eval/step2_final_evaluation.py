#!/usr/bin/env python3
"""
Step 2 — Final evaluation.

Runs ALL inference needed for the rest of the pipeline exactly once:
val, test, and counterfactual predictions for the baseline, V3-A, and
V3-C.

Every later step (3-6) reads only the files this script writes.
The visualization script (7) reads only the analysis outputs of 3-6.

Does not retrain, does not modify checkpoints, does not touch V1.

Default paths are configured for the current Socia project layout,
but can be overridden from the command line.

Usage:
    python step2_final_evaluation.py

Or with explicit paths:
    python step2_final_evaluation.py \
        --data-dir /path/to/Socia/ml/dataset/scripts/data \
        --checkpoints-dir /path/to/Socia/ml/models/checkpoints \
        --code-dir /path/to/Socia/ml/models \
        --out-dir /path/to/Socia/ml/eval/eval_outputs
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import common


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            r"D:\AI Companion\Socia\ml\dataset\scripts\data"
        ),
        help=(
            "Directory containing train.jsonl, val.jsonl, test.jsonl, "
            "and counterfactual_eval.jsonl"
        ),
    )

    p.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path(
            r"D:\AI Companion\Socia\ml\models\checkpoints"
        ),
        help=(
            "Directory containing best_model_v3a.pt "
            "and best_model_v3c.pt"
        ),
    )

    p.add_argument(
        "--code-dir",
        type=Path,
        default=Path(
            r"D:\AI Companion\Socia\ml\models"
        ),
        help=(
            "Directory containing the frozen model code"
        ),
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            r"D:\AI Companion\Socia\ml\eval\eval_outputs"
        ),
        help="Directory where Step 2 outputs will be written",
    )

    return p.parse_args()


def fit_time_normalization_v2(
    records: List[dict],
) -> Dict[str, float]:
    """
    Reproduce the exact V2 temporal normalization statistics.

    Statistics are computed from TRAIN only.

    Raw temporal quantities:
        - absolute delta_hours
        - gap_prev
        - gap_same_category

    Each quantity is transformed with log1p before calculating
    mean and population standard deviation.

    For gap_prev:
        The first event of each window is excluded because its
        gap_prev is deterministically zero.

    For gap_same_category:
        Only events with a previous occurrence of the same category
        are included. Placeholder zeros for "no previous occurrence"
        are excluded.
    """

    abs_logged: List[float] = []
    gap_prev_logged: List[float] = []
    gap_same_cat_logged: List[float] = []

    for record in records:
        events = record["events"]

        category_ids = [
            int(e["category_id"])
            for e in events
        ]

        raw_hours = [
            float(e["delta_hours"])
            for e in events
        ]

        n = len(events)

        last_seen: Dict[int, float] = {}

        for i in range(n):

            # ------------------------------------------------------
            # Absolute event time
            # ------------------------------------------------------
            abs_logged.append(
                math.log1p(raw_hours[i])
            )

            # ------------------------------------------------------
            # Gap from previous event
            #
            # First event has no previous event, so it is excluded.
            # ------------------------------------------------------
            if i > 0:
                gap = (
                    raw_hours[i]
                    - raw_hours[i - 1]
                )

                gap_prev_logged.append(
                    math.log1p(gap)
                )

            # ------------------------------------------------------
            # Gap from previous event of the same category
            #
            # Only real same-category gaps are included.
            # ------------------------------------------------------
            cat = category_ids[i]

            if cat in last_seen:
                same_cat_gap = (
                    raw_hours[i]
                    - last_seen[cat]
                )

                gap_same_cat_logged.append(
                    math.log1p(same_cat_gap)
                )

            last_seen[cat] = raw_hours[i]

    # ------------------------------------------------------------------
    # Calculate mean/std exactly from the collected log1p values.
    #
    # math.fsum is used for numerically stable summation.
    # The std here is population std (ddof=0), matching numpy.std()
    # when called without ddof.
    # ------------------------------------------------------------------

    def mean_std(values: List[float]):
        if not values:
            raise ValueError(
                "Cannot compute normalization statistics from "
                "an empty value list."
            )

        mean = math.fsum(values) / len(values)

        variance = (
            math.fsum(
                (x - mean) ** 2
                for x in values
            )
            / len(values)
        )

        std = math.sqrt(variance)

        return float(mean), float(std)

    abs_mean, abs_std = mean_std(abs_logged)
    gap_prev_mean, gap_prev_std = mean_std(
        gap_prev_logged
    )
    gap_same_cat_mean, gap_same_cat_std = mean_std(
        gap_same_cat_logged
    )

    return {
        "abs_mean": abs_mean,
        "abs_std": abs_std,
        "gap_prev_mean": gap_prev_mean,
        "gap_prev_std": gap_prev_std,
        "gap_same_cat_mean": gap_same_cat_mean,
        "gap_same_cat_std": gap_same_cat_std,
    }


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Add frozen project-code directory to sys.path.
    #
    # This is needed by common.load_model() and the frozen model files.
    # No dataset_v2 import is required here.
    # ------------------------------------------------------------------
    common.add_code_dirs_to_path(args.code_dir)

    out = args.out_dir
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ================================================================
    # LOAD DATA
    # ================================================================
    print("Loading data...")

    train_records = common.load_jsonl(
        args.data_dir / "train.jsonl"
    )

    val_records = common.load_jsonl(
        args.data_dir / "val.jsonl"
    )

    test_records = common.load_jsonl(
        args.data_dir / "test.jsonl"
    )

    cf_rows = common.load_counterfactual(
        args.data_dir / "counterfactual_eval.jsonl"
    )

    print(
        f"train={len(train_records)} "
        f"val={len(val_records)} "
        f"test={len(test_records)} "
        f"counterfactual={len(cf_rows)}"
    )

    report = {
        "models": {}
    }

    # ================================================================
    # RECOMPUTE TIME NORMALIZATION STATS
    # ================================================================
    recomputed_time_stats = fit_time_normalization_v2(
        train_records
    )

    print(
        "\nRecomputed time_stats from train.jsonl:",
        recomputed_time_stats,
    )

    # ================================================================
    # BASELINE
    # Timing + category shortcut model
    # ================================================================
    import baseline as bl

    print(
        "\nFitting timing+category baseline on train..."
    )

    bl_model, bl_cols = bl.fit_baseline(
        train_records
    )

    bl_val_probs = bl.predict_baseline(
        bl_model,
        bl_cols,
        val_records,
    )

    bl_test_probs = bl.predict_baseline(
        bl_model,
        bl_cols,
        test_records,
    )

    bl_cf_probs = bl.predict_baseline(
        bl_model,
        bl_cols,
        cf_rows,
    )

    # ---------------------------------------------------------------
    # Select threshold on validation ONLY
    # ---------------------------------------------------------------
    val_labels, val_p, val_ids = (
        common.align_predictions(
            val_records,
            bl_val_probs,
        )
    )

    thresholds = common.get_thresholds(
        0.05,
        0.95,
        0.01,
    )

    (
        bl_thr,
        bl_p_val,
        bl_r_val,
        bl_f1_val,
    ) = common.sweep_best_threshold(
        val_labels,
        val_p,
        thresholds,
    )

    # ---------------------------------------------------------------
    # Evaluate baseline on test
    # ---------------------------------------------------------------
    test_labels, test_p, test_ids = (
        common.align_predictions(
            test_records,
            bl_test_probs,
        )
    )

    bl_metrics = common.compute_full_metrics(
        test_labels,
        test_p,
        bl_thr,
    )

    bl_metrics[
        "val_threshold_source"
    ] = "recomputed_on_val_this_run"

    report["models"][
        "baseline_timing_category"
    ] = bl_metrics

    # ---------------------------------------------------------------
    # Save baseline validation predictions
    # ---------------------------------------------------------------
    common.write_jsonl(
        out / "predictions_val_baseline.jsonl",
        [
            {
                "sample_id": sid,
                "label": l,
                "prob": p,
            }
            for sid, l, p in zip(
                val_ids,
                val_labels,
                val_p,
            )
        ],
    )

    # ---------------------------------------------------------------
    # Save baseline test predictions
    # ---------------------------------------------------------------
    common.write_jsonl(
        out / "predictions_test_baseline.jsonl",
        [
            {
                "sample_id": sid,
                "label": l,
                "prob": p,
            }
            for sid, l, p in zip(
                test_ids,
                test_labels,
                test_p,
            )
        ],
    )

    # ---------------------------------------------------------------
    # Save baseline counterfactual predictions
    # ---------------------------------------------------------------
    common.write_jsonl(
        out / "predictions_counterfactual_baseline.jsonl",
        [
            {
                "sample_id": r["sample_id"],
                "counterfactual_group_id": r[
                    "counterfactual_group_id"
                ],
                "role": r["role"],
                "transform": r["transform"],
                "base_sample_id": r["base_sample_id"],
                "prob": bl_cf_probs[
                    r["sample_id"]
                ],
            }
            for r in cf_rows
        ],
    )

    print(
        f"baseline: "
        f"val_thr={bl_thr} "
        f"test={bl_metrics}"
    )

    # ================================================================
    # V3-A / V3-C
    # ================================================================
    for version in common.MODEL_VERSIONS:

        ckpt_path = (
            args.checkpoints_dir
            / f"best_model_{version}.pt"
        )

        print(
            f"\nLoading {version} from "
            f"{ckpt_path} ..."
        )

        model, meta, device = (
            common.load_model(
                version,
                ckpt_path,
                device=None,
            )
        )

        print(
            f"  checkpoint epoch={meta['epoch']} "
            f"val_f1={meta['val_f1']:.4f} "
            f"val_threshold={meta['val_threshold']} "
            f"seed={meta['seed']}"
        )

        # ------------------------------------------------------------
        # Cross-check checkpoint time statistics against train.jsonl
        # ------------------------------------------------------------
        stored_ts = meta["time_stats"]

        max_abs_diff = max(
            abs(
                stored_ts[k]
                - recomputed_time_stats[k]
            )
            for k in stored_ts
        )

        print(
            "  time_stats max abs diff vs "
            "recomputed from train.jsonl: "
            f"{max_abs_diff:.8f}"
        )

        if max_abs_diff > 1e-4:
            print(
                "  *** WARNING: time_stats mismatch "
                "exceeds tolerance. This checkpoint may "
                "not have been trained on "
                "--data-dir/train.jsonl. ***"
            )

        # IMPORTANT:
        # Use checkpoint's own stored stats for inference.
        time_stats = stored_ts

        # ------------------------------------------------------------
        # Inference exactly once per split
        # ------------------------------------------------------------
        val_probs = common.run_inference(
            model,
            val_records,
            time_stats,
            device,
        )

        test_probs = common.run_inference(
            model,
            test_records,
            time_stats,
            device,
        )

        cf_probs = common.run_inference(
            model,
            cf_rows,
            time_stats,
            device,
        )

        # ------------------------------------------------------------
        # Reproduce validation threshold
        # ------------------------------------------------------------
        val_labels, val_p, val_ids = (
            common.align_predictions(
                val_records,
                val_probs,
            )
        )

        (
            my_thr,
            my_p,
            my_r,
            my_f1,
        ) = common.sweep_best_threshold(
            val_labels,
            val_p,
            common.get_thresholds(),
        )

        print(
            f"  reproduced val threshold={my_thr} "
            f"(checkpoint stored "
            f"{meta['val_threshold']}) "
            f"reproduced val_f1={my_f1:.4f} "
            f"(checkpoint stored "
            f"{meta['val_f1']:.4f})"
        )

        if (
            abs(
                my_thr
                - float(meta["val_threshold"])
            )
            > 1e-9
            or abs(
                my_f1
                - float(meta["val_f1"])
            )
            > 1e-3
        ):
            print(
                "  *** WARNING: reproduced val "
                "threshold/F1 does not match the "
                "checkpoint's stored value. Using the "
                "freshly reproduced validation-derived "
                "threshold. No test data is used for "
                "threshold selection. ***"
            )

        # ------------------------------------------------------------
        # Evaluate on test
        # ------------------------------------------------------------
        test_labels, test_p, test_ids = (
            common.align_predictions(
                test_records,
                test_probs,
            )
        )

        metrics = common.compute_full_metrics(
            test_labels,
            test_p,
            my_thr,
        )

        metrics[
            "val_threshold_source"
        ] = "recomputed_on_val_this_run"

        metrics[
            "checkpoint_reported_val_f1"
        ] = float(meta["val_f1"])

        metrics[
            "checkpoint_reported_val_threshold"
        ] = float(meta["val_threshold"])

        metrics[
            "reproduced_val_f1"
        ] = my_f1

        metrics[
            "reproduced_val_threshold"
        ] = my_thr

        metrics[
            "time_stats_max_abs_diff_vs_recomputed"
        ] = max_abs_diff

        report["models"][version] = metrics

        # ------------------------------------------------------------
        # Save validation predictions
        # ------------------------------------------------------------
        common.write_jsonl(
            out / f"predictions_val_{version}.jsonl",
            [
                {
                    "sample_id": sid,
                    "label": l,
                    "prob": p,
                }
                for sid, l, p in zip(
                    val_ids,
                    val_labels,
                    val_p,
                )
            ],
        )

        # ------------------------------------------------------------
        # Save test predictions
        # ------------------------------------------------------------
        common.write_jsonl(
            out / f"predictions_test_{version}.jsonl",
            [
                {
                    "sample_id": sid,
                    "label": l,
                    "prob": p,
                }
                for sid, l, p in zip(
                    test_ids,
                    test_labels,
                    test_p,
                )
            ],
        )

        # ------------------------------------------------------------
        # Save counterfactual predictions
        # ------------------------------------------------------------
        common.write_jsonl(
            out / f"predictions_counterfactual_{version}.jsonl",
            [
                {
                    "sample_id": r["sample_id"],
                    "counterfactual_group_id": r[
                        "counterfactual_group_id"
                    ],
                    "role": r["role"],
                    "transform": r["transform"],
                    "base_sample_id": r["base_sample_id"],
                    "prob": cf_probs[
                        r["sample_id"]
                    ],
                }
                for r in cf_rows
            ],
        )

        print(
            f"  test metrics: {metrics}"
        )

    # ================================================================
    # Cross-model identical test set check
    # ================================================================
    id_sets = {}

    for name in [
        "baseline",
        "v3a",
        "v3c",
    ]:
        rows = common.load_jsonl(
            out / f"predictions_test_{name}.jsonl"
        )

        id_sets[name] = {
            r["sample_id"]
            for r in rows
        }

    same_set = (
        id_sets["baseline"]
        == id_sets["v3a"]
        == id_sets["v3c"]
    )

    report[
        "same_frozen_test_set_across_models"
    ] = same_set

    print(
        "\nSame frozen test set across "
        f"baseline/v3a/v3c: {same_set}"
    )

    # ================================================================
    # Write final JSON report
    # ================================================================
    common.write_json(
        out / "final_evaluation_report.json",
        report,
    )

    # ================================================================
    # Write flat CSV report
    # ================================================================
    with (
        out / "final_evaluation_report.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.writer(f)

        w.writerow(
            [
                "model",
                "n",
                "threshold",
                "roc_auc",
                "pr_auc",
                "f1",
                "precision",
                "recall",
                "tp",
                "tn",
                "fp",
                "fn",
            ]
        )

        for name, m in report["models"].items():

            w.writerow(
                [
                    name,
                    m["n"],
                    m["threshold"],
                    m["roc_auc"],
                    m["pr_auc"],
                    m["f1"],
                    m["precision"],
                    m["recall"],
                    m["tp"],
                    m["tn"],
                    m["fp"],
                    m["fn"],
                ]
            )

    print(
        f"\nWrote: "
        f"{out / 'final_evaluation_report.json'}"
    )

    print(
        f"Wrote: "
        f"{out / 'final_evaluation_report.csv'}"
    )

    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()