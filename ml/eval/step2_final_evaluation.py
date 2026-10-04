"""
Step 2 — Final evaluation on the frozen V2 benchmark.

Runs all five final comparison models:

1. Timing + category baseline
2. V2
3. V3-A
4. V3-C
5. V3-A + bucketed recurrence features

For every model:
- evaluate on the same frozen train/val/test benchmark
- use train-fitted temporal normalization
- select the operating threshold on validation only
- evaluate the frozen test set
- save validation predictions
- save test predictions
- run counterfactual inference
- save counterfactual predictions

All later evaluation steps read only the files produced here.

Does not retrain neural models and does not modify checkpoints.

Usage:
    python -m ml.eval.step2_final_evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.models.dataset import load_jsonl
from ml.models.dataset_v2 import build_datasets_v2

from .baseline import (
    fit_baseline,
    predict_baseline,
)

from . import common


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

ALL_MODEL_NAMES = [
    "baseline_timing_category",
    "v2",
    "v3a",
    "v3c",
    "v3a_bucketed_recurrence",
]

NEURAL_MODEL_NAMES = [
    "v2",
    "v3a",
    "v3c",
    "v3a_bucketed_recurrence",
]

PREDICTION_FILE_NAMES = {
    "baseline_timing_category": "baseline",
    "v2": "v2",
    "v3a": "v3a",
    "v3c": "v3c",
    "v3a_bucketed_recurrence": "v3a_bucketed_recurrence",
}


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def compute_binary_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, float]:

    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    preds = (probs >= threshold).astype(int)

    # Fixed label order:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        preds,
        labels=[0, 1],
    ).ravel()

    return {
        "roc_auc": float(
            roc_auc_score(y_true, probs)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probs)
        ),
        "f1": float(
            f1_score(
                y_true,
                preds,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                preds,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                preds,
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                preds,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }



def select_validation_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, float]:
    """
    Select the operating threshold using validation data only.

    Uses the same fixed threshold grid used by the existing evaluation
    utilities, rather than selecting from individual probability values.
    """

    thresholds = common.get_thresholds(
        0.05,
        0.95,
        0.01,
    )

    best_threshold = None
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    for threshold in thresholds:
        preds = (
            probs >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            preds,
            zero_division=0,
        )

        precision = precision_score(
            y_true,
            preds,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            preds,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
            best_precision = float(precision)
            best_recall = float(recall)

    if best_threshold is None:
        raise RuntimeError(
            "Could not select a validation threshold."
        )

    return {
        "threshold": best_threshold,
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
    }


def select_test_best_threshold_diagnostic(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, float]:
    """
    Diagnostic only.

    Selects the best threshold on the test set so that the best achievable
    test F1 can be reported separately.

    This value MUST NOT be used for model selection or primary reporting.
    """

    thresholds = common.get_thresholds(
        0.05,
        0.95,
        0.01,
    )

    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    best_threshold = None
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for threshold in thresholds:
        preds = (
            probs >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            preds,
            zero_division=0,
        )

        precision = precision_score(
            y_true,
            preds,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            preds,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
            best_precision = float(precision)
            best_recall = float(recall)

    return {
        "threshold": float(best_threshold),
        "f1": float(best_f1),
        "precision": float(best_precision),
        "recall": float(best_recall),
    }


# ---------------------------------------------------------------------
# Prediction saving
# ---------------------------------------------------------------------

def save_predictions(
    output_dir: Path,
    model_name: str,
    split: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> Path:
    """
    Save standard validation/test predictions.

    Schema:
        probability,label

    Row order is exactly the order of the corresponding frozen benchmark
    records.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_stem = PREDICTION_FILE_NAMES[
        model_name
    ]

    path = (
        output_dir
        / f"{file_stem}_{split}_predictions.csv"
    )

    df = pd.DataFrame(
        {
            "probability": np.asarray(
                probabilities,
                dtype=float,
            ),
            "label": np.asarray(
                labels,
                dtype=int,
            ),
        }
    )

    df.to_csv(
        path,
        index=False,
    )

    return path


# ---------------------------------------------------------------------
# Counterfactual prediction saving
# ---------------------------------------------------------------------

def save_counterfactual_predictions(
    output_dir: Path,
    model_name: str,
    cf_rows: List[dict],
    cf_probabilities: Dict[str, float],
) -> Path:
    """
    Save counterfactual predictions in the format expected by the
    downstream counterfactual analysis.

    Each row retains the metadata needed to reconstruct base/transformed
    pairs:

        sample_id
        counterfactual_group_id
        role
        transform
        base_sample_id
        prob
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / f"predictions_counterfactual_{PREDICTION_FILE_NAMES[model_name]}.jsonl"
    )

    rows = []

    for record in cf_rows:
        sample_id = record["sample_id"]

        if sample_id not in cf_probabilities:
            raise KeyError(
                f"Missing counterfactual probability for "
                f"sample_id={sample_id!r} "
                f"while saving {model_name}."
            )

        rows.append(
            {
                "sample_id": sample_id,
                "counterfactual_group_id": record[
                    "counterfactual_group_id"
                ],
                "role": record["role"],
                "transform": record["transform"],
                "base_sample_id": record[
                    "base_sample_id"
                ],
                "prob": float(
                    cf_probabilities[sample_id]
                ),
            }
        )

    common.write_jsonl(
        path,
        rows,
    )

    return path


# ---------------------------------------------------------------------
# Identical test-set verification
# ---------------------------------------------------------------------

def verify_identical_test_set(
    prediction_dir: Path,
    model_names: List[str],
) -> None:
    """
    Verify that all model prediction files contain the same ordered
    test labels.

    Since the CSV prediction format intentionally does not contain
    sample_id, equality of the ordered labels plus equal row counts
    verifies that Step 2 preserved the same frozen record order.
    """

    reference_labels = None
    reference_model = None

    for model_name in model_names:

        file_stem = PREDICTION_FILE_NAMES[
            model_name
        ]

        path = (
            prediction_dir
            / f"{file_stem}_test_predictions.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing test prediction file for "
                f"{model_name}: {path}"
            )

        df = pd.read_csv(path)

        if "label" not in df.columns:
            raise ValueError(
                f"Prediction file does not contain "
                f"'label': {path}"
            )

        labels = df[
            "label"
        ].to_numpy(
            dtype=int
        )

        if reference_labels is None:
            reference_labels = labels
            reference_model = model_name
            continue

        if len(labels) != len(
            reference_labels
        ):
            raise ValueError(
                "Test-set size mismatch: "
                f"{reference_model}="
                f"{len(reference_labels)}, "
                f"{model_name}="
                f"{len(labels)}"
            )

        if not np.array_equal(
            labels,
            reference_labels,
        ):
            raise ValueError(
                "Test labels/order mismatch "
                "between "
                f"{reference_model} and "
                f"{model_name}"
            )

    print(
        f"Verified identical ordered test set "
        f"across {len(model_names)} models."
    )


# ---------------------------------------------------------------------
# Counterfactual integrity checks
# ---------------------------------------------------------------------

def verify_counterfactual_rows(
    cf_rows: List[dict],
) -> None:
    """
    Verify the minimum metadata required by downstream counterfactual
    analysis.
    """

    required_fields = {
        "sample_id",
        "counterfactual_group_id",
        "role",
        "transform",
        "base_sample_id",
    }

    if not cf_rows:
        raise ValueError(
            "Counterfactual dataset is empty."
        )

    for index, row in enumerate(cf_rows):

        missing = (
            required_fields
            - set(row.keys())
        )

        if missing:
            raise ValueError(
                "Counterfactual row "
                f"{index} is missing fields: "
                f"{sorted(missing)}"
            )

    sample_ids = [
        row["sample_id"]
        for row in cf_rows
    ]

    if len(sample_ids) != len(
        set(sample_ids)
    ):
        raise ValueError(
            "Counterfactual sample_id values "
            "are not unique."
        )

    print(
        "Verified counterfactual metadata: "
        f"{len(cf_rows)} unique rows."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run final Step 2 evaluation across "
            "the timing/category baseline, V2, "
            "V3-A, V3-C, and V3-A bucketed "
            "recurrence models."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "ml/dataset/scripts/data"
        ),
        help=(
            "Directory containing the frozen "
            "V2 benchmark data."
        ),
    )

    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path(
            "ml/models/checkpoints"
        ),
        help=(
            "Directory containing neural "
            "model checkpoints."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "ml/eval/eval_outputs"
        ),
        help=(
            "Directory for final evaluation "
            "outputs."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Torch device, e.g. cuda or cpu. "
            "Defaults to automatic selection."
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_dir = (
        args.output_dir
        / "predictions"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("SOCIA — STEP 2 FINAL EVALUATION")
    print("=" * 80)

    print()
    print("Models:")
    for model_name in ALL_MODEL_NAMES:
        print(f"  - {model_name}")

    # =================================================================
    # BUILD / LOAD FROZEN BENCHMARK
    # =================================================================

    print()
    print("=" * 80)
    print("BUILDING V2 BENCHMARK")
    print("=" * 80)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        time_stats,
    ) = build_datasets_v2(
        args.data_dir
    )

    # Raw records are used directly by both the baseline and neural
    # inference code.
    train_records = load_jsonl(
        args.data_dir
        / "train.jsonl"
    )

    val_records = load_jsonl(
        args.data_dir
        / "val.jsonl"
    )

    test_records = load_jsonl(
        args.data_dir
        / "test.jsonl"
    )

    # Counterfactual benchmark is required by Step 4.
    cf_rows = common.load_counterfactual(
        args.data_dir
        / "counterfactual_eval.jsonl"
    )

    print(
        f"Train records        : "
        f"{len(train_records)}"
    )

    print(
        f"Val records          : "
        f"{len(val_records)}"
    )

    print(
        f"Test records         : "
        f"{len(test_records)}"
    )

    print(
        f"Counterfactual rows  : "
        f"{len(cf_rows)}"
    )

    verify_counterfactual_rows(
        cf_rows
    )

    # =================================================================
    # LABELS
    # =================================================================

    y_train = np.asarray(
        [
            record["label"]
            for record in train_records
        ],
        dtype=int,
    )

    y_val = np.asarray(
        [
            record["label"]
            for record in val_records
        ],
        dtype=int,
    )

    y_test = np.asarray(
        [
            record["label"]
            for record in test_records
        ],
        dtype=int,
    )

    print()
    print("Label distribution:")

    print(
        f"  Train: positive="
        f"{int(y_train.sum())}, "
        f"negative="
        f"{int(len(y_train) - y_train.sum())}"
    )

    print(
        f"  Val  : positive="
        f"{int(y_val.sum())}, "
        f"negative="
        f"{int(len(y_val) - y_val.sum())}"
    )

    print(
        f"  Test : positive="
        f"{int(y_test.sum())}, "
        f"negative="
        f"{int(len(y_test) - y_test.sum())}"
    )

    # =================================================================
    # FINAL REPORT
    # =================================================================

    report = {
        "benchmark": {
            "train_size": int(
                len(train_records)
            ),
            "val_size": int(
                len(val_records)
            ),
            "test_size": int(
                len(test_records)
            ),
            "test_positive": int(
                y_test.sum()
            ),
            "test_negative": int(
                len(y_test)
                - y_test.sum()
            ),
            "counterfactual_size": int(
                len(cf_rows)
            ),
        },
        "models": {},
    }

    # =================================================================
    # MODEL 1 — TIMING + CATEGORY BASELINE
    # =================================================================

    print()
    print("=" * 80)
    print(
        "MODEL 1 / 5 — "
        "TIMING + CATEGORY BASELINE"
    )
    print("=" * 80)

    baseline_model, baseline_cols = (
        fit_baseline(
            train_records
        )
    )

    # -------------------------------------------------------------
    # Normal inference
    # -------------------------------------------------------------

    baseline_val_prob_dict = (
        predict_baseline(
            baseline_model,
            baseline_cols,
            val_records,
        )
    )

    baseline_test_prob_dict = (
        predict_baseline(
            baseline_model,
            baseline_cols,
            test_records,
        )
    )

    baseline_cf_prob_dict = (
        predict_baseline(
            baseline_model,
            baseline_cols,
            cf_rows,
        )
    )

    # Restore exact frozen-record order.
    baseline_val_probs = np.asarray(
        [
            baseline_val_prob_dict[
                record["sample_id"]
            ]
            for record in val_records
        ],
        dtype=float,
    )

    baseline_test_probs = np.asarray(
        [
            baseline_test_prob_dict[
                record["sample_id"]
            ]
            for record in test_records
        ],
        dtype=float,
    )

    # -------------------------------------------------------------
    # Validation threshold
    # -------------------------------------------------------------

    baseline_val_selection = (
        select_validation_threshold(
            y_val,
            baseline_val_probs,
        )
    )

    baseline_threshold = (
        baseline_val_selection[
            "threshold"
        ]
    )

    baseline_val_metrics = (
        compute_binary_metrics(
            y_val,
            baseline_val_probs,
            baseline_threshold,
        )
    )

    baseline_test_metrics = (
        compute_binary_metrics(
            y_test,
            baseline_test_probs,
            baseline_threshold,
        )
    )

    baseline_test_best = (
        select_test_best_threshold_diagnostic(
            y_test,
            baseline_test_probs,
        )
    )

    print()
    print("Baseline validation:")
    print(
        f"  threshold : "
        f"{baseline_threshold:.4f}"
    )
    print(
        f"  F1        : "
        f"{baseline_val_metrics['f1']:.4f}"
    )
    print(
        f"  precision : "
        f"{baseline_val_metrics['precision']:.4f}"
    )
    print(
        f"  recall    : "
        f"{baseline_val_metrics['recall']:.4f}"
    )

    print()
    print("Baseline test:")
    print(
        f"  ROC-AUC   : "
        f"{baseline_test_metrics['roc_auc']:.4f}"
    )
    print(
        f"  PR-AUC    : "
        f"{baseline_test_metrics['pr_auc']:.4f}"
    )
    print(
        f"  F1        : "
        f"{baseline_test_metrics['f1']:.4f}"
    )
    print(
        f"  precision : "
        f"{baseline_test_metrics['precision']:.4f}"
    )
    print(
        f"  recall    : "
        f"{baseline_test_metrics['recall']:.4f}"
    )
    print(
        f"  accuracy  : "
        f"{baseline_test_metrics['accuracy']:.4f}"
    )

    print()
    print(
        "Baseline diagnostic "
        "test-best F1:"
    )
    print(
        f"  threshold : "
        f"{baseline_test_best['threshold']:.4f}"
    )
    print(
        f"  F1        : "
        f"{baseline_test_best['f1']:.4f}"
    )

    # -------------------------------------------------------------
    # Save normal predictions
    # -------------------------------------------------------------

    save_predictions(
        prediction_dir,
        "baseline_timing_category",
        "val",
        baseline_val_probs,
        y_val,
    )

    save_predictions(
        prediction_dir,
        "baseline_timing_category",
        "test",
        baseline_test_probs,
        y_test,
    )

    # -------------------------------------------------------------
    # Save counterfactual predictions
    # -------------------------------------------------------------

    save_counterfactual_predictions(
        prediction_dir,
        "baseline_timing_category",
        cf_rows,
        baseline_cf_prob_dict,
    )

    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    report["models"][
        "baseline_timing_category"
    ] = {
        "type": "baseline",
        "threshold": float(
            baseline_threshold
        ),
        "validation": baseline_val_metrics,
        "test": baseline_test_metrics,
        "test_best_threshold_diagnostic": (
            baseline_test_best
        ),
    }

    # =================================================================
    # MODELS 2–5 — NEURAL MODELS
    # =================================================================

    for index, version in enumerate(
        NEURAL_MODEL_NAMES,
        start=2,
    ):

        print()
        print("=" * 80)
        print(
            f"MODEL {index} / 5 — "
            f"{version.upper()}"
        )
        print("=" * 80)

        checkpoint_path = (
            args.checkpoints_dir
            / f"best_model_{version}.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for "
                f"{version}: "
                f"{checkpoint_path}"
            )

        print(
            f"Checkpoint: "
            f"{checkpoint_path}"
        )

        # -------------------------------------------------------------
        # Load model
        # -------------------------------------------------------------

        model, meta, device = (
            common.load_model(
                version,
                checkpoint_path,
                device=args.device,
            )
        )

        print(
            f"Device: {device}"
        )

        if meta:
            print(
                "Checkpoint metadata:"
            )
            print(meta)

        # -------------------------------------------------------------
        # Cross-check checkpoint time stats
        # -------------------------------------------------------------

        if meta and "time_stats" in meta:

            stored_time_stats = (
                meta["time_stats"]
            )

            max_abs_diff = max(
                abs(
                    float(
                        stored_time_stats[key]
                    )
                    - float(
                        time_stats[key]
                    )
                )
                for key in stored_time_stats
            )

            print(
                "Time stats max abs diff "
                "vs current train-fitted "
                f"stats: {max_abs_diff:.8f}"
            )

            if max_abs_diff > 1e-4:
                print(
                    "  *** WARNING: checkpoint "
                    "time_stats differ from "
                    "train.jsonl-derived stats "
                    "by more than 1e-4. ***"
                )

        # -------------------------------------------------------------
        # Validation inference
        # -------------------------------------------------------------

        val_probs = common.run_inference(
            model,
            val_records,
            time_stats,
            device,
            version=version,
        )

        # -------------------------------------------------------------
        # Test inference
        # -------------------------------------------------------------

        test_probs = common.run_inference(
            model,
            test_records,
            time_stats,
            device,
            version=version,
        )

        # -------------------------------------------------------------
        # Counterfactual inference
        #
        # IMPORTANT:
        # This is required by Step 4.
        # -------------------------------------------------------------

        cf_probs = common.run_inference(
            model,
            cf_rows,
            time_stats,
            device,
            version=version,
        )

        # -------------------------------------------------------------
        # Validation threshold
        # -------------------------------------------------------------

        val_selection = (
            select_validation_threshold(
                y_val,
                val_probs,
            )
        )

        threshold = (
            val_selection["threshold"]
        )

        val_metrics = (
            compute_binary_metrics(
                y_val,
                val_probs,
                threshold,
            )
        )

        test_metrics = (
            compute_binary_metrics(
                y_test,
                test_probs,
                threshold,
            )
        )

        # -------------------------------------------------------------
        # Test-best diagnostic
        # -------------------------------------------------------------

        test_best = (
            select_test_best_threshold_diagnostic(
                y_test,
                test_probs,
            )
        )

        # -------------------------------------------------------------
        # Print
        # -------------------------------------------------------------

        print()
        print(
            f"{version} validation:"
        )

        print(
            f"  threshold : "
            f"{threshold:.4f}"
        )

        print(
            f"  F1        : "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"  precision : "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"  recall    : "
            f"{val_metrics['recall']:.4f}"
        )

        print()
        print(
            f"{version} test:"
        )

        print(
            f"  ROC-AUC   : "
            f"{test_metrics['roc_auc']:.4f}"
        )

        print(
            f"  PR-AUC    : "
            f"{test_metrics['pr_auc']:.4f}"
        )

        print(
            f"  F1        : "
            f"{test_metrics['f1']:.4f}"
        )

        print(
            f"  precision : "
            f"{test_metrics['precision']:.4f}"
        )

        print(
            f"  recall    : "
            f"{test_metrics['recall']:.4f}"
        )

        print(
            f"  accuracy  : "
            f"{test_metrics['accuracy']:.4f}"
        )

        print()
        print(
            f"{version} diagnostic "
            "test-best F1:"
        )

        print(
            f"  threshold : "
            f"{test_best['threshold']:.4f}"
        )

        print(
            f"  F1        : "
            f"{test_best['f1']:.4f}"
        )

        # -------------------------------------------------------------
        # Save normal predictions
        # -------------------------------------------------------------

        save_predictions(
            prediction_dir,
            version,
            "val",
            val_probs,
            y_val,
        )

        save_predictions(
            prediction_dir,
            version,
            "test",
            test_probs,
            y_test,
        )

        # -------------------------------------------------------------
        # Save counterfactual predictions
        # -------------------------------------------------------------

        cf_probability_dict = {
            row["sample_id"]: float(
                cf_probs[index]
            )
            for index, row in enumerate(
                cf_rows
            )
        }

        save_counterfactual_predictions(
            prediction_dir,
            version,
            cf_rows,
            cf_probability_dict,
        )

        # -------------------------------------------------------------
        # Store report
        # -------------------------------------------------------------

        model_report = {
            "type": "neural",
            "checkpoint": str(
                checkpoint_path
            ),
            "threshold": float(
                threshold
            ),
            "validation": val_metrics,
            "test": test_metrics,
            "test_best_threshold_diagnostic": (
                test_best
            ),
        }

        if meta:
            model_report[
                "checkpoint_metadata"
            ] = meta

        report["models"][version] = (
            model_report
        )

    # =================================================================
    # VERIFY STANDARD TEST PREDICTIONS
    # =================================================================

    print()
    print("=" * 80)
    print(
        "VERIFYING IDENTICAL TEST SET "
        "ACROSS ALL FIVE MODELS"
    )
    print("=" * 80)

    verify_identical_test_set(
        prediction_dir,
        ALL_MODEL_NAMES,
    )

    # =================================================================
    # VERIFY COUNTERFACTUAL OUTPUTS
    # =================================================================

    print()
    print("=" * 80)
    print(
        "VERIFYING COUNTERFACTUAL OUTPUTS"
    )
    print("=" * 80)

    for model_name in ALL_MODEL_NAMES:

        file_stem = (
            PREDICTION_FILE_NAMES[
                model_name
            ]
        )

        cf_path = (
            prediction_dir
            / f"predictions_counterfactual_{file_stem}.jsonl"
        )

        if not cf_path.exists():
            raise FileNotFoundError(
                f"Missing counterfactual "
                f"prediction file for "
                f"{model_name}: "
                f"{cf_path}"
            )

        saved_cf_rows = (
            common.load_jsonl(
                cf_path
            )
        )

        if len(saved_cf_rows) != len(
            cf_rows
        ):
            raise ValueError(
                "Counterfactual row-count "
                "mismatch for "
                f"{model_name}: "
                f"expected {len(cf_rows)}, "
                f"got {len(saved_cf_rows)}"
            )

        expected_ids = [
            row["sample_id"]
            for row in cf_rows
        ]

        actual_ids = [
            row["sample_id"]
            for row in saved_cf_rows
        ]

        if actual_ids != expected_ids:
            raise ValueError(
                "Counterfactual sample order "
                "mismatch for "
                f"{model_name}."
            )

        print(
            f"  {model_name}: "
            f"{len(saved_cf_rows)} rows OK"
        )

    # =================================================================
    # SAVE FINAL JSON REPORT
    # =================================================================

    report_path = (
        args.output_dir
        / "final_evaluation_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # =================================================================
    # SAVE COMPACT CSV COMPARISON
    # =================================================================

    rows = []

    for model_name in ALL_MODEL_NAMES:

        model_report = (
            report["models"][
                model_name
            ]
        )

        test_metrics = (
            model_report["test"]
        )

        rows.append(
            {
                "model": model_name,
                "threshold": (
                    model_report[
                        "threshold"
                    ]
                ),
                "roc_auc": (
                    test_metrics[
                        "roc_auc"
                    ]
                ),
                "pr_auc": (
                    test_metrics[
                        "pr_auc"
                    ]
                ),
                "f1": (
                    test_metrics[
                        "f1"
                    ]
                ),
                "precision": (
                    test_metrics[
                        "precision"
                    ]
                ),
                "recall": (
                    test_metrics[
                        "recall"
                    ]
                ),
                "accuracy": (
                    test_metrics[
                        "accuracy"
                    ]
                ),
            }
        )

    comparison_df = pd.DataFrame(
        rows
    )

    comparison_path = (
        args.output_dir
        / "final_evaluation_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print()
    print("=" * 80)
    print(
        "FINAL FIVE-MODEL COMPARISON"
    )
    print("=" * 80)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print("=" * 80)
    print("OUTPUTS")
    print("=" * 80)

    print(
        f"Report : "
        f"{report_path}"
    )

    print(
        f"CSV    : "
        f"{comparison_path}"
    )

    print(
        f"Predictions: "
        f"{prediction_dir}"
    )

    print()
    print(
        "Step 2 completed successfully."
    )

    print(
        "All five models were evaluated "
        "on the same frozen test set."
    )

    print(
        "Validation, test, and "
        "counterfactual predictions "
        "were saved for downstream "
        "evaluation steps."
    )


if __name__ == "__main__":
    main()