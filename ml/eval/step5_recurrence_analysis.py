"""
Step 5 — Recurrence-count analysis.

Uses:
- test.jsonl
- test_eval_metadata.jsonl
- final model predictions from Step 2
- frozen thresholds from final_evaluation_report.json

Analyzes whether model probability / recall changes with the
intended number of recurrence sites.

Important:
- n_occ_intended comes from test_eval_metadata.jsonl.
- Only positive examples are included in recurrence-count analysis.
- n_occ_intended is never inferred from the event sequence.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# step5_recurrence_analysis.py lives in:
#   ml/eval/
#
# Therefore:
#   ML_DIR   = ml/
#   EVAL_DIR = ml/eval/eval_outputs/
#
ML_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(
    ML_DIR,
    "dataset",
    "scripts",
    "data",
)

EVAL_DIR = os.path.join(
    SCRIPT_DIR,
    "eval_outputs",
)

PRED_DIR = os.path.join(
    EVAL_DIR,
    "predictions",
)


MODEL_FILES = {
    "baseline_timing_category":
        "baseline_test_predictions.csv",

    "v2":
        "v2_test_predictions.csv",

    "v3a":
        "v3a_test_predictions.csv",

    "v3c":
        "v3c_test_predictions.csv",

    "v3a_bucketed_recurrence":
        "v3a_bucketed_recurrence_test_predictions.csv",
}

MODEL_ORDER = [
    "baseline_timing_category",
    "v2",
    "v3a",
    "v3c",
    "v3a_bucketed_recurrence",
]


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def load_json(path: str) -> Any:
    require_file(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    require_file(path)

    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: "
                    f"{exc}"
                ) from exc

    return rows


def load_csv(path: str) -> List[Dict[str, str]]:
    require_file(path)

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV has no header: {path}"
            )

        return list(reader)


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_csv(
    path: str,
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric value for {field_name}: {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"Non-finite numeric value for {field_name}: {value!r}"
        )

    return result


def safe_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid integer value for {field_name}: {value!r}"
        ) from exc

    return result


# ---------------------------------------------------------------------
# Load / validate test data
# ---------------------------------------------------------------------

def load_test_records() -> List[Dict[str, Any]]:
    path = os.path.join(
        DATA_DIR,
        "test.jsonl",
    )

    rows = load_jsonl(path)

    required = {
        "events",
        "label",
        "sample_id",
        "user_id",
    }

    for index, row in enumerate(rows):
        missing = sorted(required - set(row.keys()))

        if missing:
            raise ValueError(
                f"test.jsonl row {index} is missing fields: {missing}"
            )

    return rows


def load_test_metadata() -> Dict[str, Dict[str, Any]]:
    path = os.path.join(
        DATA_DIR,
        "test_eval_metadata.jsonl",
    )

    rows = load_jsonl(path)

    required = {
        "sample_id",
        "split",
        "label",
        "subtype",
        "n_occ_intended",
    }

    metadata_by_id: Dict[str, Dict[str, Any]] = {}

    for index, row in enumerate(rows):
        missing = sorted(required - set(row.keys()))

        if missing:
            raise ValueError(
                f"test_eval_metadata.jsonl row {index} "
                f"is missing fields: {missing}"
            )

        sample_id = str(row["sample_id"])

        if sample_id in metadata_by_id:
            raise ValueError(
                f"Duplicate sample_id in test_eval_metadata.jsonl: "
                f"{sample_id}"
            )

        metadata_by_id[sample_id] = row

    return metadata_by_id


def join_test_records_with_metadata(
    test_records: List[Dict[str, Any]],
    metadata_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:

    test_ids = {
        str(row["sample_id"])
        for row in test_records
    }

    metadata_ids = set(metadata_by_id.keys())

    missing_metadata = sorted(test_ids - metadata_ids)
    extra_metadata = sorted(metadata_ids - test_ids)

    if missing_metadata:
        raise ValueError(
            "test_eval_metadata.jsonl is missing metadata for "
            f"{len(missing_metadata)} test samples. "
            f"First IDs: {missing_metadata[:10]}"
        )

    if extra_metadata:
        raise ValueError(
            "test_eval_metadata.jsonl contains "
            f"{len(extra_metadata)} IDs not present in test.jsonl. "
            f"First IDs: {extra_metadata[:10]}"
        )

    joined: List[Dict[str, Any]] = []

    for row in test_records:
        sample_id = str(row["sample_id"])
        metadata = metadata_by_id[sample_id]

        test_label = safe_int(
            row["label"],
            "test label",
        )

        metadata_label = safe_int(
            metadata["label"],
            "metadata label",
        )

        if test_label != metadata_label:
            raise ValueError(
                f"Label mismatch for sample_id={sample_id}: "
                f"test.jsonl={test_label}, "
                f"metadata={metadata_label}"
            )

        joined.append(
            {
                **row,
                "subtype": str(metadata["subtype"]),
                "n_occ_intended": safe_int(
                    metadata["n_occ_intended"],
                    "n_occ_intended",
                ),
            }
        )

    return joined


# ---------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------

def load_predictions(
    model_name: str,
) -> Tuple[List[float], List[int]]:

    filename = MODEL_FILES[model_name]

    path = os.path.join(
        PRED_DIR,
        filename,
    )

    rows = load_csv(path)

    required = {
        "probability",
        "label",
    }

    if not rows:
        raise ValueError(
            f"Prediction file is empty: {path}"
        )

    if not required.issubset(rows[0].keys()):
        raise ValueError(
            f"Prediction file {path} is missing required fields. "
            f"Expected at least {sorted(required)}, "
            f"got {list(rows[0].keys())}"
        )

    probabilities: List[float] = []
    labels: List[int] = []

    for index, row in enumerate(rows):
        probability = safe_float(
            row["probability"],
            f"{model_name} probability row {index}",
        )

        label = safe_int(
            row["label"],
            f"{model_name} label row {index}",
        )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Probability outside [0,1] for "
                f"{model_name}, row {index}: {probability}"
            )

        if label not in (0, 1):
            raise ValueError(
                f"Invalid label for {model_name}, row {index}: {label}"
            )

        probabilities.append(probability)
        labels.append(label)

    return probabilities, labels


# ---------------------------------------------------------------------
# Final evaluation report / thresholds
# ---------------------------------------------------------------------

def load_final_report() -> Dict[str, Any]:
    path = os.path.join(
        EVAL_DIR,
        "final_evaluation_report.json",
    )

    require_file(path)

    return load_json(path)


def get_model_threshold(
    report: Dict[str, Any],
    model_name: str,
) -> float:

    models = report.get("models")

    if not isinstance(models, dict):
        raise ValueError(
            "final_evaluation_report.json does not contain "
            "a valid 'models' object."
        )

    if model_name not in models:
        raise ValueError(
            f"Model {model_name!r} not found in final evaluation report."
        )

    model_info = models[model_name]

    if "threshold" not in model_info:
        raise ValueError(
            f"Model {model_name!r} has no threshold "
            "in final evaluation report."
        )

    threshold = safe_float(
        model_info["threshold"],
        f"{model_name} threshold",
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Invalid threshold for {model_name}: {threshold}"
        )

    return threshold


# ---------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------

def rank_average(values: List[float]) -> List[float]:
    indexed = sorted(
        enumerate(values),
        key=lambda x: x[1],
    )

    ranks = [0.0] * len(values)

    i = 0

    while i < len(indexed):
        j = i + 1

        while (
            j < len(indexed)
            and indexed[j][1] == indexed[i][1]
        ):
            j += 1

        average_rank = (i + 1 + j) / 2.0

        for k in range(i, j):
            original_index = indexed[k][0]
            ranks[original_index] = average_rank

        i = j

    return ranks


def pearson_correlation(
    x: List[float],
    y: List[float],
) -> float:

    if len(x) != len(y):
        raise ValueError(
            "Pearson inputs must have equal length."
        )

    if len(x) < 2:
        return float("nan")

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = sum(
        (a - mean_x) * (b - mean_y)
        for a, b in zip(x, y)
    )

    denom_x = math.sqrt(
        sum((a - mean_x) ** 2 for a in x)
    )

    denom_y = math.sqrt(
        sum((b - mean_y) ** 2 for b in y)
    )

    if denom_x == 0.0 or denom_y == 0.0:
        return float("nan")

    return numerator / (denom_x * denom_y)


def spearman_correlation(
    x: List[float],
    y: List[float],
) -> float:

    if len(x) != len(y):
        raise ValueError(
            "Spearman inputs must have equal length."
        )

    if len(x) < 2:
        return float("nan")

    return pearson_correlation(
        rank_average(x),
        rank_average(y),
    )


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 78)
    print("SOCIA — STEP 5: RECURRENCE-COUNT ANALYSIS")
    print("=" * 78)

    print(f"Data directory : {DATA_DIR}")
    print(f"Eval directory : {EVAL_DIR}")
    print(f"Prediction dir : {PRED_DIR}")
    print()

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    test_records = load_test_records()

    print(
        f"Loaded test records: {len(test_records)}"
    )

    metadata_by_id = load_test_metadata()

    print(
        f"Loaded test metadata: {len(metadata_by_id)}"
    )

    joined_records = join_test_records_with_metadata(
        test_records,
        metadata_by_id,
    )

    print(
        f"Joined test + metadata: {len(joined_records)}"
    )

    # -------------------------------------------------------------
    # Positive-only recurrence analysis
    # -------------------------------------------------------------

    positive_records = [
        row
        for row in joined_records
        if safe_int(row["label"], "label") == 1
    ]

    if not positive_records:
        raise ValueError(
            "No positive examples found in test set."
        )

    print(
        f"Positive examples for recurrence analysis: "
        f"{len(positive_records)}"
    )

    # Make sure the positive test examples really are the intended
    # positive subtype, rather than negative subtypes with n_occ_intended
    # values such as 1 or 2.
    unexpected_positive_subtypes = sorted(
        {
            row["subtype"]
            for row in positive_records
            if row["subtype"] != "positive"
        }
    )

    if unexpected_positive_subtypes:
        raise ValueError(
            "Found label=1 examples with non-positive subtype: "
            f"{unexpected_positive_subtypes}"
        )

    # -------------------------------------------------------------
    # Load final evaluation report
    # -------------------------------------------------------------

    report = load_final_report()

    thresholds = {
        model_name: get_model_threshold(
            report,
            model_name,
        )
        for model_name in MODEL_ORDER
    }

    print("\nFrozen thresholds:")

    for model_name in MODEL_ORDER:
        print(
            f"  {model_name:32s}: "
            f"{thresholds[model_name]:.4f}"
        )

    # -------------------------------------------------------------
    # Load predictions
    # -------------------------------------------------------------

    all_predictions: Dict[str, List[float]] = {}

    for model_name in MODEL_ORDER:

        probabilities, labels = load_predictions(
            model_name
        )

        if len(probabilities) != len(test_records):
            raise ValueError(
                f"{model_name} prediction count mismatch: "
                f"{len(probabilities)} predictions vs "
                f"{len(test_records)} test records."
            )

        expected_labels = [
            safe_int(row["label"], "test label")
            for row in test_records
        ]

        if labels != expected_labels:
            mismatch_index = next(
                (
                    i
                    for i, (a, b)
                    in enumerate(
                        zip(labels, expected_labels)
                    )
                    if a != b
                ),
                None,
            )

            raise ValueError(
                f"{model_name} prediction labels do not align "
                f"with test.jsonl. First mismatch index: "
                f"{mismatch_index}"
            )

        all_predictions[model_name] = probabilities

    # -------------------------------------------------------------
    # Build positive-index list
    # -------------------------------------------------------------

    positive_indices = [
        index
        for index, row in enumerate(test_records)
        if safe_int(row["label"], "label") == 1
    ]

    if len(positive_indices) != len(positive_records):
        raise RuntimeError(
            "Internal positive-index mismatch."
        )

    # -------------------------------------------------------------
    # Group by intended recurrence count
    # -------------------------------------------------------------

    recurrence_buckets = defaultdict(list)

    for index in positive_indices:

        record = joined_records[index]

        n_occ = safe_int(
            record["n_occ_intended"],
            "n_occ_intended",
        )

        if n_occ < 2:
            raise ValueError(
                "Positive example has n_occ_intended < 2: "
                f"sample_id={record['sample_id']}, "
                f"n_occ_intended={n_occ}"
            )

        recurrence_buckets[n_occ].append(index)

    bucket_values = sorted(
        recurrence_buckets.keys()
    )

    print("\nRecurrence-count buckets:")

    for n_occ in bucket_values:
        print(
            f"  n_occ_intended={n_occ}: "
            f"{len(recurrence_buckets[n_occ])} examples"
        )

    # -------------------------------------------------------------
    # Per-model analysis
    # -------------------------------------------------------------

    analysis: Dict[str, Any] = {
        "metadata": {
            "description": (
                "Positive-only recurrence-count analysis using "
                "n_occ_intended from test_eval_metadata.jsonl."
            ),
            "source_test_file": os.path.join(
                DATA_DIR,
                "test.jsonl",
            ),
            "source_metadata_file": os.path.join(
                DATA_DIR,
                "test_eval_metadata.jsonl",
            ),
            "test_size": len(test_records),
            "positive_count": len(positive_records),
            "models": MODEL_ORDER,
            "thresholds": thresholds,
        },
        "models": {},
    }

    csv_rows: List[Dict[str, Any]] = []

    for model_name in MODEL_ORDER:

        probabilities = all_predictions[model_name]
        threshold = thresholds[model_name]

        model_buckets: Dict[str, Any] = {}

        bucket_probabilities: List[float] = []
        bucket_recurrence_counts: List[float] = []

        print("\n" + "-" * 78)
        print(model_name)
        print("-" * 78)

        for n_occ in bucket_values:

            indices = recurrence_buckets[n_occ]

            probs = [
                probabilities[index]
                for index in indices
            ]

            mean_probability = (
                sum(probs) / len(probs)
            )

            sorted_probs = sorted(probs)

            middle = len(sorted_probs) // 2

            if len(sorted_probs) % 2 == 0:
                median_probability = (
                    sorted_probs[middle - 1]
                    + sorted_probs[middle]
                ) / 2.0
            else:
                median_probability = sorted_probs[middle]

            positive_predictions = sum(
                probability >= threshold
                for probability in probs
            )

            recall = (
                positive_predictions / len(probs)
            )

            bucket_key = str(n_occ)

            model_buckets[bucket_key] = {
                "n": len(probs),
                "mean_probability": mean_probability,
                "median_probability": median_probability,
                "recall_at_threshold": recall,
                "threshold": threshold,
            }

            bucket_probabilities.append(
                mean_probability
            )

            bucket_recurrence_counts.append(
                float(n_occ)
            )

            csv_rows.append(
                {
                    "model": model_name,
                    "n_occ_intended": n_occ,
                    "n": len(probs),
                    "mean_probability": mean_probability,
                    "median_probability": median_probability,
                    "recall_at_threshold": recall,
                    "threshold": threshold,
                }
            )

            print(
                f"n={n_occ:2d} | "
                f"N={len(probs):4d} | "
                f"mean={mean_probability:.4f} | "
                f"median={median_probability:.4f} | "
                f"recall={recall:.4f}"
            )

        # ---------------------------------------------------------
        # Monotonicity
        # ---------------------------------------------------------

        means = [
            model_buckets[str(n)]["mean_probability"]
            for n in bucket_values
        ]

        recalls = [
            model_buckets[str(n)]["recall_at_threshold"]
            for n in bucket_values
        ]

        probability_monotonic = all(
            means[i] <= means[i + 1] + 1e-12
            for i in range(len(means) - 1)
        )

        recall_monotonic = all(
            recalls[i] <= recalls[i + 1] + 1e-12
            for i in range(len(recalls) - 1)
        )

        # ---------------------------------------------------------
        # Spearman correlations
        # ---------------------------------------------------------

        probability_spearman = spearman_correlation(
            bucket_recurrence_counts,
            bucket_probabilities,
        )

        recall_spearman = spearman_correlation(
            bucket_recurrence_counts,
            recalls,
        )

        model_analysis = {
            "threshold": threshold,
            "buckets": model_buckets,
            "monotonicity": {
                "mean_probability_non_decreasing":
                    probability_monotonic,
                "recall_non_decreasing":
                    recall_monotonic,
            },
            "spearman": {
                "recurrence_count_vs_mean_probability":
                    probability_spearman,
                "recurrence_count_vs_recall":
                    recall_spearman,
            },
        }

        analysis["models"][model_name] = model_analysis

    # -------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------

    json_path = os.path.join(
        EVAL_DIR,
        "recurrence_analysis.json",
    )

    csv_path = os.path.join(
        EVAL_DIR,
        "recurrence_analysis.csv",
    )

    write_json(
        json_path,
        analysis,
    )

    write_csv(
        csv_path,
        csv_rows,
        [
            "model",
            "n_occ_intended",
            "n",
            "mean_probability",
            "median_probability",
            "recall_at_threshold",
            "threshold",
        ],
    )

    print("\n" + "=" * 78)
    print("STEP 5 COMPLETE")
    print("=" * 78)

    print(
        f"JSON: {json_path}"
    )

    print(
        f"CSV : {csv_path}"
    )


if __name__ == "__main__":
    main()

