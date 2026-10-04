"""
Step 4 — Counterfactual analysis.

This step is aligned with the current Step 2 and common.py.

Step 2 writes counterfactual predictions as JSONL files:

    predictions_counterfactual_baseline.jsonl
    predictions_counterfactual_v2.jsonl
    predictions_counterfactual_v3a.jsonl
    predictions_counterfactual_v3c.jsonl
    predictions_counterfactual_v3a_bucketed_recurrence.jsonl

Each prediction row contains:

    sample_id
    counterfactual_group_id
    role
    transform
    base_sample_id
    prob

No model inference happens here.

The counterfactual benchmark contains base positive examples and
transformed versions of those examples.

For every transformed example:

    change = counterfactual_probability - original_probability

Because the transformations are designed to disrupt recurrence-related
structure, the expected direction is a probability decrease.

For each model × transform, this step reports:

    - N
    - mean original probability
    - median original probability
    - mean counterfactual probability
    - median counterfactual probability
    - mean probability change
    - median probability change
    - fraction moving in the expected direction (change < 0)

It also writes one pair-level CSV per model.

Outputs:

    eval_outputs/counterfactual_analysis.json
    eval_outputs/counterfactual_analysis.csv

    eval_outputs/counterfactual_pairs_baseline_timing_category.csv
    eval_outputs/counterfactual_pairs_v2.csv
    eval_outputs/counterfactual_pairs_v3a.csv
    eval_outputs/counterfactual_pairs_v3c.csv
    eval_outputs/counterfactual_pairs_v3a_bucketed_recurrence.csv

Usage:

    python -m ml.eval.step4_counterfactual_analysis
"""

from __future__ import annotations

import argparse
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
        help=(
            "Directory containing counterfactual_eval.jsonl."
        ),
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            r"D:\AI Companion\Socia\ml\eval\eval_outputs"
        ),
        help=(
            "Directory containing Step 2 prediction outputs "
            "and where Step 4 outputs will be written."
        ),
    )

    return p.parse_args()


# =====================================================================
# COUNTERFACTUAL DATASET
# =====================================================================

def load_counterfactual_dataset(
    path: Path,
) -> list[dict]:
    """
    Load the frozen counterfactual evaluation dataset.

    The dataset is used as an integrity/reference source. Prediction
    probabilities themselves come from the Step 2 JSONL outputs.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Counterfactual evaluation dataset not found: {path}"
        )

    rows = common.load_jsonl(path)

    if not rows:
        raise ValueError(
            f"Counterfactual evaluation dataset is empty: {path}"
        )

    return rows


# =====================================================================
# STEP 2 COUNTERFACTUAL PREDICTIONS
# =====================================================================

def load_counterfactual_predictions(
    path: Path,
) -> list[dict]:
    """
    Load one counterfactual prediction JSONL produced by Step 2.

    Current Step 2 schema:

        sample_id
        counterfactual_group_id
        role
        transform
        base_sample_id
        prob
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Counterfactual prediction file not found: {path}\n"
            "Run Step 2 first."
        )

    rows = common.load_jsonl(path)

    if not rows:
        raise ValueError(
            f"Counterfactual prediction file is empty: {path}"
        )

    required_fields = {
        "sample_id",
        "counterfactual_group_id",
        "role",
        "transform",
        "base_sample_id",
        "prob",
    }

    for index, row in enumerate(rows):

        missing = (
            required_fields
            - set(row.keys())
        )

        if missing:
            raise ValueError(
                f"Counterfactual prediction row {index} "
                f"is missing required field(s): "
                f"{sorted(missing)}\n"
                f"File: {path}\n"
                f"Available fields: {sorted(row.keys())}"
            )

        try:
            probability = float(
                row["prob"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid probability at row {index} "
                f"in {path}: {row['prob']!r}"
            ) from exc

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Probability outside [0, 1] at row {index} "
                f"in {path}: {probability}"
            )

    return rows


# =====================================================================
# VALIDATION
# =====================================================================

def validate_prediction_dataset_alignment(
    dataset_rows: list[dict],
    prediction_rows: list[dict],
    model_key: str,
) -> None:
    """
    Validate that a Step 2 counterfactual prediction file corresponds
    exactly to the current counterfactual evaluation dataset.

    Current Step 2 preserves the counterfactual dataset row order, so
    we validate both row count and identity/order.
    """

    if len(dataset_rows) != len(prediction_rows):
        raise ValueError(
            f"Counterfactual row-count mismatch for "
            f"'{model_key}':\n"
            f"  dataset rows:     {len(dataset_rows)}\n"
            f"  prediction rows:  {len(prediction_rows)}"
        )

    identity_fields = [
        "sample_id",
        "counterfactual_group_id",
        "role",
        "transform",
        "base_sample_id",
    ]

    mismatches = []

    for index, (
        dataset_row,
        prediction_row,
    ) in enumerate(
        zip(
            dataset_rows,
            prediction_rows,
        )
    ):

        for field in identity_fields:

            dataset_value = dataset_row.get(
                field
            )

            prediction_value = prediction_row.get(
                field
            )

            if dataset_value != prediction_value:
                mismatches.append(
                    {
                        "row_index": index,
                        "field": field,
                        "dataset_value": dataset_value,
                        "prediction_value": prediction_value,
                    }
                )

                break

    if mismatches:

        preview = mismatches[:10]

        raise ValueError(
            f"Counterfactual dataset/prediction identity mismatch "
            f"for '{model_key}'.\n"
            f"Number of mismatched rows: {len(mismatches)}\n"
            f"First mismatches: {preview}\n\n"
            "The Step 2 counterfactual prediction file does not "
            "match the current counterfactual_eval.jsonl."
        )


# =====================================================================
# STRUCTURAL VALIDATION
# =====================================================================

def validate_counterfactual_structure(
    rows: list[dict],
    model_key: str,
) -> None:
    """
    Validate base/transformed structure.

    Every transformed row must reference a base sample that exists in
    the same prediction file.
    """

    base_rows = [
        row
        for row in rows
        if row["role"] == "base"
    ]

    transformed_rows = [
        row
        for row in rows
        if row["role"] == "transformed"
    ]

    unexpected_roles = sorted(
        {
            row["role"]
            for row in rows
            if row["role"]
            not in {"base", "transformed"}
        }
    )

    if unexpected_roles:
        raise ValueError(
            f"Unexpected counterfactual role(s) for "
            f"'{model_key}': {unexpected_roles}"
        )

    base_by_id = {}

    for row in base_rows:

        sample_id = row["sample_id"]

        if sample_id in base_by_id:
            raise ValueError(
                f"Duplicate base sample_id "
                f"'{sample_id}' for model '{model_key}'."
            )

        base_by_id[sample_id] = row

    if not base_by_id:
        raise ValueError(
            f"No base rows found for model '{model_key}'."
        )

    if not transformed_rows:
        raise ValueError(
            f"No transformed rows found for model '{model_key}'."
        )

    for row in transformed_rows:

        transformed_sample_id = row[
            "sample_id"
        ]

        base_sample_id = row[
            "base_sample_id"
        ]

        if base_sample_id not in base_by_id:
            raise ValueError(
                f"Transformed sample "
                f"'{transformed_sample_id}' references base "
                f"sample '{base_sample_id}', but that base sample "
                f"does not exist for model '{model_key}'."
            )

    # ---------------------------------------------------------------
    # Each transformed row must have a transform.
    # Base rows are allowed to carry their generator-provided
    # transform value, but transformed rows must never be empty.
    # ---------------------------------------------------------------

    for row in transformed_rows:

        transform = row["transform"]

        if transform is None or str(transform).strip() == "":
            raise ValueError(
                f"Transformed sample '{row['sample_id']}' "
                f"has an empty transform for model "
                f"'{model_key}'."
            )


# =====================================================================
# PAIR CONSTRUCTION
# =====================================================================

def build_pairs(
    rows: list[dict],
    model_key: str,
) -> pd.DataFrame:
    """
    Build base → counterfactual probability pairs.

    Pairing is performed using base_sample_id, not positional offsets.
    """

    base_probs: dict[str, float] = {}

    for row in rows:

        if row["role"] != "base":
            continue

        sample_id = row["sample_id"]

        if sample_id in base_probs:
            raise ValueError(
                f"Duplicate base sample_id "
                f"'{sample_id}' for model '{model_key}'."
            )

        base_probs[sample_id] = float(
            row["prob"]
        )

    pairs = []

    for row in rows:

        if row["role"] != "transformed":
            continue

        transformed_sample_id = row[
            "sample_id"
        ]

        base_sample_id = row[
            "base_sample_id"
        ]

        if base_sample_id not in base_probs:
            raise KeyError(
                f"Transformed sample "
                f"'{transformed_sample_id}' references "
                f"missing base sample '{base_sample_id}'."
            )

        original_prob = base_probs[
            base_sample_id
        ]

        counterfactual_prob = float(
            row["prob"]
        )

        change = (
            counterfactual_prob
            - original_prob
        )

        pairs.append(
            {
                "model": model_key,
                "transform": row[
                    "transform"
                ],
                "counterfactual_group_id": row[
                    "counterfactual_group_id"
                ],
                "base_sample_id": base_sample_id,
                "counterfactual_sample_id": (
                    transformed_sample_id
                ),
                "original_prob": original_prob,
                "counterfactual_prob": (
                    counterfactual_prob
                ),
                "change": change,
            }
        )

    if not pairs:
        raise ValueError(
            f"No base/transformed pairs constructed "
            f"for model '{model_key}'."
        )

    return pd.DataFrame(pairs)


# =====================================================================
# AGGREGATION
# =====================================================================

def aggregate_by_transform(
    pairs_df: pd.DataFrame,
    model_key: str,
) -> list[dict]:
    """
    Aggregate counterfactual shifts by transform.

    Expected direction is a decrease:

        counterfactual_prob < original_prob

    Therefore:

        expected_direction = change < 0
    """

    rows = []

    for transform, sub in pairs_df.groupby(
        "transform",
        sort=True,
    ):

        n = len(sub)

        if n == 0:
            continue

        change = sub[
            "change"
        ]

        fraction_expected = float(
            (change < 0).mean()
        )

        rows.append(
            {
                "model": model_key,
                "transform": transform,
                "n": int(n),
                "mean_original_prob": float(
                    sub[
                        "original_prob"
                    ].mean()
                ),
                "median_original_prob": float(
                    sub[
                        "original_prob"
                    ].median()
                ),
                "mean_counterfactual_prob": float(
                    sub[
                        "counterfactual_prob"
                    ].mean()
                ),
                "median_counterfactual_prob": float(
                    sub[
                        "counterfactual_prob"
                    ].median()
                ),
                "mean_change": float(
                    change.mean()
                ),
                "median_change": float(
                    change.median()
                ),
                (
                    "fraction_moving_expected_"
                    "direction_decrease"
                ): fraction_expected,
            }
        )

    if not rows:
        raise ValueError(
            f"No transform-level results generated "
            f"for model '{model_key}'."
        )

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
    # Load the current counterfactual evaluation dataset.
    # ---------------------------------------------------------------

    cf_dataset_path = (
        data_dir
        / "counterfactual_eval.jsonl"
    )

    cf_dataset = load_counterfactual_dataset(
        cf_dataset_path
    )

    print(
        "Loaded counterfactual dataset rows: "
        f"{len(cf_dataset)}"
    )

    # ---------------------------------------------------------------
    # Analyze all five canonical models.
    # ---------------------------------------------------------------

    all_rows: list[dict] = []
    per_model_pairs: dict[str, pd.DataFrame] = {}

    for model_key in MODELS:

        print("\n" + "=" * 78)
        print(
            f"MODEL: {model_key}"
        )
        print("=" * 78)

        model_filename = FILE_NAME_FOR_MODEL[
            model_key
        ]

        prediction_path = (
            predictions_dir
            / (
                "predictions_counterfactual_"
                f"{model_filename}.jsonl"
            )
        )

        print(
            f"Predictions: {prediction_path}"
        )

        # -----------------------------------------------------------
        # Load exact Step 2 JSONL output.
        # -----------------------------------------------------------

        prediction_rows = (
            load_counterfactual_predictions(
                prediction_path
            )
        )

        print(
            "Prediction rows: "
            f"{len(prediction_rows)}"
        )

        # -----------------------------------------------------------
        # Validate against current counterfactual dataset.
        # -----------------------------------------------------------

        validate_prediction_dataset_alignment(
            dataset_rows=cf_dataset,
            prediction_rows=prediction_rows,
            model_key=model_key,
        )

        # -----------------------------------------------------------
        # Validate base/transformed structure.
        # -----------------------------------------------------------

        validate_counterfactual_structure(
            rows=prediction_rows,
            model_key=model_key,
        )

        # -----------------------------------------------------------
        # Build base → transformed probability pairs.
        # -----------------------------------------------------------

        pairs_df = build_pairs(
            rows=prediction_rows,
            model_key=model_key,
        )

        per_model_pairs[
            model_key
        ] = pairs_df

        print(
            f"Constructed pairs: {len(pairs_df)}"
        )

        # -----------------------------------------------------------
        # Aggregate by transform.
        # -----------------------------------------------------------

        model_results = aggregate_by_transform(
            pairs_df=pairs_df,
            model_key=model_key,
        )

        all_rows.extend(
            model_results
        )

    # ---------------------------------------------------------------
    # Build final aggregate DataFrame.
    # ---------------------------------------------------------------

    out_df = pd.DataFrame(
        all_rows
    )

    if out_df.empty:
        raise ValueError(
            "Counterfactual analysis produced no results."
        )

    out_df = out_df.sort_values(
        [
            "transform",
            "model",
        ]
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------------
    # Save aggregate CSV.
    # ---------------------------------------------------------------

    csv_path = (
        out_dir
        / "counterfactual_analysis.csv"
    )

    out_df.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save aggregate JSON.
    # ---------------------------------------------------------------

    json_path = (
        out_dir
        / "counterfactual_analysis.json"
    )

    common.write_json(
        json_path,
        all_rows,
    )

    # ---------------------------------------------------------------
    # Save pair-level CSV for every model.
    # ---------------------------------------------------------------

    for model_key, pair_df in (
        per_model_pairs.items()
    ):

        pair_path = (
            out_dir
            / (
                "counterfactual_pairs_"
                f"{model_key}.csv"
            )
        )

        pair_df.to_csv(
            pair_path,
            index=False,
        )

        print(
            f"Wrote: {pair_path}"
        )

    # ---------------------------------------------------------------
    # Print summary.
    # ---------------------------------------------------------------

    print("\n" + "=" * 78)
    print("COUNTERFACTUAL ANALYSIS COMPLETE")
    print("=" * 78)

    print(
        out_df.to_string(
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

