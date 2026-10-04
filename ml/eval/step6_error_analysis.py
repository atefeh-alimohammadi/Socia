"""
Step 6 — Error analysis.

For each selected model, breaks down:
- false positives by negative subtype
- false negatives by intended recurrence-count bucket

The analysis uses:
- test_eval_metadata.jsonl for subtype and n_occ_intended
- the frozen test prediction CSVs produced by Step 2
- final_evaluation_report.json for the frozen validation-selected thresholds

No inference happens here.

Current Step 2 prediction format:
    <model>_test_predictions.csv

with columns:
    probability,label

Usage:
    python -m ml.eval.step6_error_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import common


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

MODELS = [
    "baseline_timing_category",
    "v3a",
    "v3c",
]

FILE_NAME_FOR_MODEL = {
    "baseline_timing_category": "baseline_test_predictions.csv",
    "v3a": "v3a_test_predictions.csv",
    "v3c": "v3c_test_predictions.csv",
}


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent

DEFAULT_DATA_DIR = (
    ML_DIR
    / "dataset"
    / "scripts"
    / "data"
)

DEFAULT_OUT_DIR = (
    SCRIPT_DIR
    / "eval_outputs"
)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Directory containing test.jsonl and "
            "test_eval_metadata.jsonl."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            "Step 2 evaluation output directory containing "
            "predictions/, final_evaluation_report.json, "
            "and where Step 6 outputs will be written."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def load_prediction_csv(
    path: Path,
    expected_n: int,
) -> pd.DataFrame:

    require_file(path)

    df = pd.read_csv(path)

    required_columns = {
        "probability",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Prediction file is missing required column(s): "
            f"{sorted(missing)}\n"
            f"File: {path}\n"
            f"Available columns: {list(df.columns)}"
        )

    if len(df) != expected_n:
        raise ValueError(
            f"Prediction count mismatch in {path}: "
            f"expected {expected_n}, got {len(df)}"
        )

    if df["probability"].isna().any():
        raise ValueError(
            f"Prediction file contains NaN probabilities: {path}"
        )

    if df["label"].isna().any():
        raise ValueError(
            f"Prediction file contains NaN labels: {path}"
        )

    df["probability"] = pd.to_numeric(
        df["probability"],
        errors="raise",
    )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int)

    if ((df["probability"] < 0.0) |
            (df["probability"] > 1.0)).any():
        raise ValueError(
            f"Prediction probabilities must be in [0, 1]: {path}"
        )

    invalid_labels = set(df["label"].unique()) - {0, 1}

    if invalid_labels:
        raise ValueError(
            f"Invalid labels in {path}: {sorted(invalid_labels)}"
        )

    return df


def load_test_records(
    data_dir: Path,
) -> list[dict]:

    path = data_dir / "test.jsonl"

    records = common.load_jsonl(path)

    if not records:
        raise ValueError(
            f"test.jsonl is empty: {path}"
        )

    required = {
        "events",
        "label",
        "sample_id",
        "user_id",
    }

    for index, row in enumerate(records):
        missing = required - set(row.keys())

        if missing:
            raise ValueError(
                f"test.jsonl row {index} is missing "
                f"required field(s): {sorted(missing)}"
            )

    return records


def validate_prediction_labels(
    df: pd.DataFrame,
    test_records: list[dict],
    model_name: str,
) -> None:

    expected_labels = [
        int(row["label"])
        for row in test_records
    ]

    actual_labels = df["label"].tolist()

    if actual_labels != expected_labels:
        mismatch_index = None

        for index, (actual, expected) in enumerate(
            zip(actual_labels, expected_labels)
        ):
            if actual != expected:
                mismatch_index = index
                break

        raise ValueError(
            f"{model_name} prediction labels do not align with "
            f"test.jsonl.\n"
            f"First mismatch index: {mismatch_index}"
        )


def build_metadata_frame(
    data_dir: Path,
    test_records: list[dict],
) -> pd.DataFrame:

    metadata_path = (
        data_dir / "test_eval_metadata.jsonl"
    )

    require_file(metadata_path)

    metadata = common.load_eval_metadata_by_id(
        metadata_path
    )

    rows = []

    for row in test_records:
        sample_id = str(row["sample_id"])

        if sample_id not in metadata:
            raise ValueError(
                f"Missing metadata for sample_id={sample_id}"
            )

        meta = metadata[sample_id]

        if "subtype" not in meta:
            raise ValueError(
                f"Metadata for sample_id={sample_id} "
                "does not contain 'subtype'."
            )

        if "n_occ_intended" not in meta:
            raise ValueError(
                f"Metadata for sample_id={sample_id} "
                "does not contain 'n_occ_intended'."
            )

        metadata_label = int(meta["label"])
        test_label = int(row["label"])

        if metadata_label != test_label:
            raise ValueError(
                f"Label mismatch for sample_id={sample_id}: "
                f"test.jsonl={test_label}, "
                f"metadata={metadata_label}"
            )

        rows.append(
            {
                "sample_id": sample_id,
                "subtype": str(meta["subtype"]),
                "n_occ_intended": int(
                    meta["n_occ_intended"]
                ),
            }
        )

    metadata_df = pd.DataFrame(rows)

    if len(metadata_df) != len(test_records):
        raise ValueError(
            "Metadata frame size does not match test set."
        )

    return metadata_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    pred_dir = out_dir / "predictions"

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("SOCIA — STEP 6: ERROR ANALYSIS")
    print("=" * 78)
    print()
    print(f"Data directory : {data_dir}")
    print(f"Eval directory : {out_dir}")
    print(f"Prediction dir : {pred_dir}")
    print()

    # -------------------------------------------------------------
    # Load frozen test records
    # -------------------------------------------------------------

    test_records = load_test_records(
        data_dir
    )

    n_test = len(test_records)

    print(
        f"Loaded test records: {n_test}"
    )

    # -------------------------------------------------------------
    # Load metadata
    # -------------------------------------------------------------

    metadata_df = build_metadata_frame(
        data_dir,
        test_records,
    )

    print(
        f"Loaded evaluation metadata: "
        f"{len(metadata_df)} rows"
    )

    # -------------------------------------------------------------
    # Load final evaluation report
    # -------------------------------------------------------------

    final_report_path = (
        out_dir / "final_evaluation_report.json"
    )

    require_file(final_report_path)

    with final_report_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        final_eval = json.load(f)

    if "models" not in final_eval:
        raise ValueError(
            "final_evaluation_report.json does not contain "
            "'models'."
        )

    # -------------------------------------------------------------
    # Load predictions and perform analysis
    # -------------------------------------------------------------

    summary = {}

    for model_key in MODELS:

        if model_key not in FILE_NAME_FOR_MODEL:
            raise ValueError(
                f"No prediction filename configured for "
                f"model {model_key!r}."
            )

        if model_key not in final_eval["models"]:
            raise ValueError(
                f"Model {model_key!r} not found in "
                "final_evaluation_report.json."
            )

        model_info = final_eval["models"][model_key]

        if "threshold" not in model_info:
            raise ValueError(
                f"No threshold found for model "
                f"{model_key!r} in final evaluation report."
            )

        threshold = float(
            model_info["threshold"]
        )

        prediction_path = (
            pred_dir
            / FILE_NAME_FOR_MODEL[model_key]
        )

        df = load_prediction_csv(
            prediction_path,
            expected_n=n_test,
        )

        validate_prediction_labels(
            df,
            test_records,
            model_key,
        )

        # ---------------------------------------------------------
        # Attach metadata by frozen test row order.
        #
        # Step 2 CSVs intentionally contain only:
        # probability,label
        #
        # Therefore sample identity comes from the frozen
        # test.jsonl order, which Step 2 validated.
        # ---------------------------------------------------------

        df["sample_id"] = [
            str(row["sample_id"])
            for row in test_records
        ]

        df["subtype"] = metadata_df[
            "subtype"
        ].values

        df["n_occ_intended"] = metadata_df[
            "n_occ_intended"
        ].values

        df["pred_positive"] = (
            df["probability"] >= threshold
        ).astype(int)

        # ---------------------------------------------------------
        # Confusion subsets
        # ---------------------------------------------------------

        fp = df[
            (df["label"] == 0)
            & (df["pred_positive"] == 1)
        ].copy()

        fn = df[
            (df["label"] == 1)
            & (df["pred_positive"] == 0)
        ].copy()

        negatives = int(
            (df["label"] == 0).sum()
        )

        positives = int(
            (df["label"] == 1).sum()
        )

        # ---------------------------------------------------------
        # False positives by negative subtype
        # ---------------------------------------------------------

        fp_by_subtype = (
            fp["subtype"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        if len(fp):
            fp_by_subtype_share = (
                fp["subtype"]
                .value_counts(normalize=True)
                .round(4)
                .sort_index()
                .to_dict()
            )
        else:
            fp_by_subtype_share = {}

        # ---------------------------------------------------------
        # False negatives by recurrence count
        #
        # n_occ_intended is only interpreted for positive
        # examples here. Negative subtypes can also have values
        # such as 1 or 2, but they do not represent recurrence
        # under the benchmark definition.
        # ---------------------------------------------------------

        if len(fn):
            fn_by_bucket = (
                fn["n_occ_intended"]
                .value_counts()
                .sort_index()
                .to_dict()
            )

            fn_by_bucket_share = (
                fn["n_occ_intended"]
                .value_counts(normalize=True)
                .round(4)
                .sort_index()
                .to_dict()
            )
        else:
            fn_by_bucket = {}
            fn_by_bucket_share = {}

        # ---------------------------------------------------------
        # Build summary
        # ---------------------------------------------------------

        summary[model_key] = {
            "threshold_used": threshold,

            "total_n": int(len(df)),

            "total_fp": int(len(fp)),
            "total_fn": int(len(fn)),

            "negative_n": negatives,
            "positive_n": positives,

            "fp_rate_overall": float(
                len(fp) / max(negatives, 1)
            ),

            "fn_rate_overall": float(
                len(fn) / max(positives, 1)
            ),

            "fp_count_by_negative_subtype": {
                str(k): int(v)
                for k, v in fp_by_subtype.items()
            },

            "fp_share_of_all_fps_by_negative_subtype": {
                str(k): float(v)
                for k, v in fp_by_subtype_share.items()
            },

            "fn_count_by_n_occ_intended": {
                str(k): int(v)
                for k, v in fn_by_bucket.items()
            },

            "fn_share_of_all_fns_by_n_occ_intended": {
                str(k): float(v)
                for k, v in fn_by_bucket_share.items()
            },
        }

        # ---------------------------------------------------------
        # Console output
        # ---------------------------------------------------------

        print()
        print(
            f"=== {model_key} "
            f"(threshold={threshold:.4f}) ==="
        )

        print(
            f"Total FP: {len(fp)} / {negatives} negatives "
            f"({summary[model_key]['fp_rate_overall']:.3f})"
        )

        print(
            f"Total FN: {len(fn)} / {positives} positives "
            f"({summary[model_key]['fn_rate_overall']:.3f})"
        )

        print(
            "FP share by negative subtype:",
            summary[model_key][
                "fp_share_of_all_fps_by_negative_subtype"
            ],
        )

        print(
            "FN share by n_occ_intended:  ",
            summary[model_key][
                "fn_share_of_all_fns_by_n_occ_intended"
            ],
        )

    # -------------------------------------------------------------
    # Save summary JSON
    # -------------------------------------------------------------

    output_json = (
        out_dir / "error_analysis.json"
    )

    common.write_json(
        output_json,
        summary,
    )

    # -------------------------------------------------------------
    # Save flat CSV
    #
    # One row per model/subtype/bucket/metric combination.
    # This is convenient for later visualization.
    # -------------------------------------------------------------

    csv_rows = []

    for model_key in MODELS:

        model_summary = summary[model_key]

        # FP subtype rows
        for subtype, count in (
            model_summary[
                "fp_count_by_negative_subtype"
            ].items()
        ):
            csv_rows.append(
                {
                    "model": model_key,
                    "error_type": "false_positive",
                    "category": subtype,
                    "count": int(count),
                    "share": float(
                        model_summary[
                            "fp_share_of_all_fps_by_negative_subtype"
                        ].get(subtype, 0.0)
                    ),
                }
            )

        # FN recurrence-count rows
        for bucket, count in (
            model_summary[
                "fn_count_by_n_occ_intended"
            ].items()
        ):
            csv_rows.append(
                {
                    "model": model_key,
                    "error_type": "false_negative",
                    "category": str(bucket),
                    "count": int(count),
                    "share": float(
                        model_summary[
                            "fn_share_of_all_fns_by_n_occ_intended"
                        ].get(bucket, 0.0)
                    ),
                }
            )

    output_csv = (
        out_dir / "error_analysis.csv"
    )

    pd.DataFrame(
        csv_rows,
        columns=[
            "model",
            "error_type",
            "category",
            "count",
            "share",
        ],
    ).to_csv(
        output_csv,
        index=False,
    )

    # -------------------------------------------------------------
    # Complete
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("STEP 6 COMPLETE")
    print("=" * 78)
    print()
    print(f"Wrote: {output_json}")
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()

