"""
Step 3 — Subtype analysis.

This step is aligned with the current Step 2 and common.py.

Current Step 2 writes test prediction CSVs with only:

    probability
    label

The current frozen V2 test.jsonl does NOT contain a subtype field and
Step 2 does NOT create test_eval_metadata.jsonl.

Therefore, subtype identity is reconstructed from the frozen V2 test-set
composition and row order:

    positive                    1000
    timing_matched               220
    order_permutation            200
    boundary_single_occurrence   200
    category_identity            160
    boundary_tight_burst         120
    pure_background              100
                                ----
                                2000

The reconstruction is only accepted when the test dataset contains
exactly 2000 rows and the labels match the expected positive/negative
layout.

No model inference happens here.

For each model × subtype, computes:

    - N
    - positive_n
    - negative_n
    - false-positive count
    - false-positive rate
    - mean predicted probability
    - median predicted probability
    - percent classified positive at the validation-selected threshold
    - positive recall for the positive subtype

Models:

    - baseline_timing_category
    - v2
    - v3a
    - v3c
    - v3a_bucketed_recurrence

Outputs:

    eval_outputs/subtype_analysis.json
    eval_outputs/subtype_analysis.csv

Usage:

    python -m ml.eval.step3_subtype_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import common


# =====================================================================
# MODEL CONFIGURATION
# =====================================================================

MODELS = [
    "baseline_timing_category",
    "v2",
    "v3a",
    "v3c",
    "v3a_bucketed_recurrence",
]


FILE_NAME_FOR_MODEL = {
    "baseline_timing_category": "baseline",
    "v2": "v2",
    "v3a": "v3a",
    "v3c": "v3c",
    "v3a_bucketed_recurrence": "v3a_bucketed_recurrence",
}


# =====================================================================
# FROZEN V2 TEST COMPOSITION
# =====================================================================
#
# IMPORTANT:
# The current test.jsonl does not store subtype.
#
# The frozen V2 test set is ordered as:
#
#   positive
#   timing_matched
#   order_permutation
#   boundary_single_occurrence
#   category_identity
#   boundary_tight_burst
#   pure_background
#
# with the following exact counts.
#
# These values must remain synchronized with the frozen V2 benchmark.
# =====================================================================

TEST_SUBTYPE_COUNTS = [
    ("positive", 1000),
    ("timing_matched", 220),
    ("order_permutation", 200),
    ("boundary_single_occurrence", 200),
    ("category_identity", 160),
    ("boundary_tight_burst", 120),
    ("pure_background", 100),
]


SUBTYPES = [
    name
    for name, _ in TEST_SUBTYPE_COUNTS
]


EXPECTED_TEST_SIZE = sum(
    count
    for _, count in TEST_SUBTYPE_COUNTS
)


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            r"D:\AI Companion\Socia\ml\dataset\scripts\data"
        ),
        help="Directory containing the frozen V2 train/val/test JSONL files.",
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            r"D:\AI Companion\Socia\ml\eval\eval_outputs"
        ),
        help=(
            "Directory containing Step 2 prediction outputs "
            "and where Step 3 outputs will be written."
        ),
    )

    return p.parse_args()


# =====================================================================
# DATA LOADING
# =====================================================================

def load_test_records(path: Path) -> list[dict]:
    """
    Load the current frozen V2 test.jsonl.

    The current schema is expected to contain:

        events
        label
        sample_id
        user_id

    Subtype is intentionally NOT expected here.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {path}"
        )

    rows = common.load_jsonl(path)

    if not rows:
        raise ValueError(
            f"Test dataset is empty: {path}"
        )

    required_fields = {
        "events",
        "label",
        "sample_id",
        "user_id",
    }

    missing = required_fields - set(rows[0].keys())

    if missing:
        raise ValueError(
            "Current test.jsonl is missing required field(s): "
            f"{sorted(missing)}\n"
            f"Available fields: {sorted(rows[0].keys())}"
        )

    return rows


# =====================================================================
# SUBTYPE RECONSTRUCTION
# =====================================================================

def build_subtype_sequence(
    test_records: list[dict],
) -> list[str]:
    """
    Reconstruct subtype labels from the frozen V2 test ordering.

    This is deliberately strict.

    We do NOT attempt to infer subtype from event content because that
    would duplicate dataset-generation logic and could silently create
    incorrect labels.

    Instead, the frozen test composition and ordering are treated as
    benchmark metadata.

    Returns one subtype string per test record, in exact row order.
    """

    n = len(test_records)

    if n != EXPECTED_TEST_SIZE:
        raise ValueError(
            "Unexpected frozen V2 test-set size.\n"
            f"Expected: {EXPECTED_TEST_SIZE}\n"
            f"Found:    {n}\n\n"
            "The Step 3 subtype reconstruction depends on the frozen "
            "V2 test composition. If the test dataset was regenerated "
            "with a different composition/order, update the frozen "
            "composition constants rather than silently analyzing it."
        )

    subtype_sequence: list[str] = []

    for subtype, count in TEST_SUBTYPE_COUNTS:
        subtype_sequence.extend(
            [subtype] * count
        )

    if len(subtype_sequence) != n:
        raise AssertionError(
            "Internal error: subtype sequence length does not match "
            "test-set length."
        )

    # ---------------------------------------------------------------
    # Validate that the reconstructed subtype labels agree with the
    # actual binary labels stored in test.jsonl.
    #
    # Positive block must have label 1.
    # Every negative subtype must have label 0.
    # ---------------------------------------------------------------

    expected_labels = [
        1 if subtype == "positive" else 0
        for subtype in subtype_sequence
    ]

    actual_labels = []

    for i, row in enumerate(test_records):
        try:
            label = int(row["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid label at test row {i}: "
                f"{row.get('label')!r}"
            ) from exc

        if label not in (0, 1):
            raise ValueError(
                f"Test row {i} has non-binary label: {label}"
            )

        actual_labels.append(label)

    mismatches = [
        i
        for i, (actual, expected) in enumerate(
            zip(actual_labels, expected_labels)
        )
        if actual != expected
    ]

    if mismatches:
        preview = mismatches[:10]

        details = []
        for i in preview:
            details.append(
                {
                    "row_index": i,
                    "sample_id": test_records[i].get("sample_id"),
                    "actual_label": actual_labels[i],
                    "expected_label": expected_labels[i],
                    "expected_subtype": subtype_sequence[i],
                }
            )

        raise ValueError(
            "Frozen V2 subtype reconstruction does not match the "
            "labels in test.jsonl.\n"
            f"Number of mismatches: {len(mismatches)}\n"
            f"First mismatches: {details}\n\n"
            "This means the current test.jsonl is not in the expected "
            "frozen V2 ordering/composition. Do not continue with "
            "subtype analysis until the dataset composition is verified."
        )

    return subtype_sequence


# =====================================================================
# PREDICTION LOADING
# =====================================================================

def load_prediction_csv(path: Path) -> pd.DataFrame:
    """
    Load one prediction CSV written by Step 2.

    Current Step 2 schema:

        probability
        label

    Rows are intentionally preserved in their original order.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction CSV not found: {path}\n"
            "Run Step 2 first."
        )

    df = pd.read_csv(path)

    required_columns = {
        "probability",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Prediction file is missing required column(s): "
            f"{sorted(missing)}\n"
            f"File: {path}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["probability"] = pd.to_numeric(
        df["probability"],
        errors="raise",
    )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int)

    if df["label"].isin([0, 1]).all() is False:
        raise ValueError(
            f"Prediction file contains non-binary labels: {path}"
        )

    if not df["probability"].between(0.0, 1.0).all():
        raise ValueError(
            f"Prediction file contains probabilities outside [0, 1]: "
            f"{path}"
        )

    return df


# =====================================================================
# FINAL REPORT
# =====================================================================

def load_final_report(path: Path) -> dict:
    """
    Load and minimally validate final_evaluation_report.json.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Final evaluation report not found: {path}\n"
            "Run Step 2 first."
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    if "models" not in report:
        raise ValueError(
            f"Invalid final evaluation report: "
            f"'models' key not found in {path}"
        )

    return report


# =====================================================================
# THRESHOLD
# =====================================================================

def get_model_threshold(
    final_eval: dict,
    model_key: str,
) -> float:
    """
    Get the validation-selected/frozen threshold recorded by Step 2.
    """

    if model_key not in final_eval["models"]:
        raise KeyError(
            f"Model '{model_key}' is missing from "
            "final_evaluation_report.json"
        )

    model_eval = final_eval["models"][model_key]

    if "threshold" not in model_eval:
        raise KeyError(
            f"Model '{model_key}' has no 'threshold' "
            "in final_evaluation_report.json"
        )

    threshold = float(
        model_eval["threshold"]
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Invalid threshold for '{model_key}': {threshold}"
        )

    return threshold


# =====================================================================
# ALIGNMENT VALIDATION
# =====================================================================

def validate_prediction_alignment(
    preds: pd.DataFrame,
    test_records: list[dict],
    model_key: str,
) -> None:
    """
    Validate positional alignment between Step 2 predictions and
    frozen test.jsonl.

    Step 2 CSVs do not contain sample_id, so row-order alignment is
    required.

    The label column provides an additional integrity check.
    """

    prediction_count = len(preds)
    test_count = len(test_records)

    if prediction_count != test_count:
        raise ValueError(
            f"Prediction/test row-count mismatch for '{model_key}':\n"
            f"  predictions: {prediction_count}\n"
            f"  test rows:   {test_count}\n\n"
            "Step 3 relies on positional alignment because the current "
            "Step 2 prediction CSVs do not contain sample_id."
        )

    expected_labels = [
        int(row["label"])
        for row in test_records
    ]

    actual_labels = (
        preds["label"]
        .astype(int)
        .tolist()
    )

    mismatches = [
        i
        for i, (actual, expected) in enumerate(
            zip(actual_labels, expected_labels)
        )
        if actual != expected
    ]

    if mismatches:
        preview = mismatches[:10]

        details = []

        for i in preview:
            details.append(
                {
                    "row_index": i,
                    "sample_id": test_records[i].get("sample_id"),
                    "prediction_label": actual_labels[i],
                    "test_label": expected_labels[i],
                }
            )

        raise ValueError(
            f"Prediction/test label mismatch for '{model_key}'.\n"
            f"Number of mismatches: {len(mismatches)}\n"
            f"First mismatches: {details}\n\n"
            "The prediction CSV is not aligned with the current "
            "test.jsonl."
        )


# =====================================================================
# SUBTYPE ANALYSIS
# =====================================================================

def analyze_model(
    model_key: str,
    preds: pd.DataFrame,
    subtype_sequence: list[str],
    threshold: float,
) -> list[dict]:
    """
    Compute subtype-level metrics for one model.
    """

    if len(preds) != len(subtype_sequence):
        raise ValueError(
            f"Internal alignment error for '{model_key}': "
            f"{len(preds)} predictions vs "
            f"{len(subtype_sequence)} subtype labels."
        )

    work = preds.copy()

    work["subtype"] = subtype_sequence

    work["pred_positive"] = (
        work["probability"] >= threshold
    ).astype(int)

    rows: list[dict] = []

    for subtype in SUBTYPES:

        sub = work[
            work["subtype"] == subtype
        ]

        if len(sub) == 0:
            raise ValueError(
                f"Expected subtype '{subtype}' but found no rows "
                f"for model '{model_key}'."
            )

        n = len(sub)

        pred_positive_count = int(
            sub["pred_positive"].sum()
        )

        pct_classified_positive = float(
            pred_positive_count / n
        )

        positive_n = int(
            (sub["label"] == 1).sum()
        )

        negative_n = int(
            (sub["label"] == 0).sum()
        )

        row = {
            "model": model_key,
            "subtype": subtype,
            "n": n,
            "positive_n": positive_n,
            "negative_n": negative_n,
            "mean_pred_prob": float(
                sub["probability"].mean()
            ),
            "median_pred_prob": float(
                sub["probability"].median()
            ),
            "pct_classified_positive_at_threshold": (
                pct_classified_positive
            ),
            "threshold_used": threshold,
        }

        if subtype == "positive":

            true_positive_count = pred_positive_count

            row["true_positive_count"] = (
                true_positive_count
            )

            row["recall_within_positive"] = float(
                true_positive_count / n
            )

            row["false_positive_count"] = 0
            row["false_positive_rate"] = 0.0

        else:

            false_positive_count = pred_positive_count

            row["false_positive_count"] = (
                false_positive_count
            )

            row["false_positive_rate"] = float(
                false_positive_count / n
            )

            row["true_positive_count"] = 0
            row["recall_within_positive"] = None

        rows.append(row)

    return rows


# =====================================================================
# MAIN
# =====================================================================

def main():
    args = parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_dir = (
        out_dir / "predictions"
    )

    if not predictions_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: "
            f"{predictions_dir}\n"
            "Run Step 2 first."
        )

    # ---------------------------------------------------------------
    # Load frozen V2 test dataset
    # ---------------------------------------------------------------

    test_path = (
        data_dir / "test.jsonl"
    )

    test_records = load_test_records(
        test_path
    )

    print(
        f"Loaded frozen V2 test records: {len(test_records)}"
    )

    # ---------------------------------------------------------------
    # Reconstruct subtype sequence from frozen V2 composition
    # ---------------------------------------------------------------

    subtype_sequence = build_subtype_sequence(
        test_records
    )

    print("\nFrozen V2 test composition:")

    for subtype, count in TEST_SUBTYPE_COUNTS:
        print(
            f"  {subtype:<30} {count}"
        )

    # ---------------------------------------------------------------
    # Load final Step 2 report
    # ---------------------------------------------------------------

    final_report_path = (
        out_dir / "final_evaluation_report.json"
    )

    final_eval = load_final_report(
        final_report_path
    )

    # ---------------------------------------------------------------
    # Analyze every canonical model
    # ---------------------------------------------------------------

    all_rows: list[dict] = []

    for model_key in MODELS:

        print("\n" + "=" * 78)
        print(
            f"MODEL: {model_key}"
        )
        print("=" * 78)

        fname = FILE_NAME_FOR_MODEL[
            model_key
        ]

        prediction_path = (
            predictions_dir
            / f"{fname}_test_predictions.csv"
        )

        print(
            f"Predictions: {prediction_path}"
        )

        preds = load_prediction_csv(
            prediction_path
        )

        print(
            f"Prediction rows: {len(preds)}"
        )

        # -----------------------------------------------------------
        # Validate positional alignment
        # -----------------------------------------------------------

        validate_prediction_alignment(
            preds=preds,
            test_records=test_records,
            model_key=model_key,
        )

        # -----------------------------------------------------------
        # Get validation-selected threshold from Step 2 report
        # -----------------------------------------------------------

        threshold = get_model_threshold(
            final_eval=final_eval,
            model_key=model_key,
        )

        print(
            f"Threshold: {threshold:.4f}"
        )

        # -----------------------------------------------------------
        # Analyze subtype behavior
        # -----------------------------------------------------------

        model_rows = analyze_model(
            model_key=model_key,
            preds=preds,
            subtype_sequence=subtype_sequence,
            threshold=threshold,
        )

        all_rows.extend(
            model_rows
        )

    # ---------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------

    out_df = pd.DataFrame(
        all_rows
    )

    csv_path = (
        out_dir / "subtype_analysis.csv"
    )

    out_df.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------

    json_path = (
        out_dir / "subtype_analysis.json"
    )

    common.write_json(
        json_path,
        all_rows,
    )

    # ---------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 78)
    print("SUBTYPE ANALYSIS COMPLETE")
    print("=" * 78)

    display_columns = [
        "model",
        "subtype",
        "n",
        "mean_pred_prob",
        "median_pred_prob",
        "pct_classified_positive_at_threshold",
        "false_positive_count",
        "false_positive_rate",
        "recall_within_positive",
    ]

    print(
        out_df[display_columns].to_string(
            index=False
        )
    )

    print(
        f"\nWrote: {json_path}"
    )

    print(
        f"Wrote: {csv_path}"
    )


if __name__ == "__main__":
    main()

