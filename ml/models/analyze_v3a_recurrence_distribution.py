"""
Socia — V3-A Recurrence Distribution Diagnostic

Purpose
-------
Analyze whether V3-A's strong dependency on
gap_same_category is caused by:

1. A genuinely strong recurrence signal in the dataset, or
2. An overly dominant / poorly represented continuous recurrence feature.

This script DOES NOT modify:
    - dataset_v2.py
    - model_v3a.py
    - checkpoints
    - train/val/test splits

It uses the exact V2 temporal-feature construction and train-fitted
normalization statistics.

Outputs
-------
1. Dataset-level recurrence statistics
2. Positive vs negative recurrence statistics
3. Recurrence coverage
4. Raw gap distributions
5. Normalized gap distributions
6. Quantiles
7. Bucket distributions
8. Positive/negative enrichment by bucket
9. Per-window recurrence density
10. Simple recurrence-only diagnostic AUC
11. Recommendations for possible bucket boundaries
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from dataset_v2 import (
    DEFAULT_DATA_DIR,
    build_datasets_v2,
)


# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = DEFAULT_DATA_DIR

# Candidate buckets are intentionally broad.
# We will NOT assume these are the final buckets.
CANDIDATE_BUCKETS = [
    ("NO_PRIOR", 0.0, 0.0),
    ("0-6h", 0.0, 6.0),
    ("6-24h", 6.0, 24.0),
    ("24-72h", 24.0, 72.0),
    ("72-168h", 72.0, 168.0),
    ("168-336h", 168.0, 336.0),
    (">336h", 336.0, float("inf")),
]

PERCENTILES = [
    1,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    99,
]


# ============================================================================
# HELPERS
# ============================================================================

def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_std(values: List[float]) -> float:
    return float(np.std(values)) if values else float("nan")


def safe_median(values: List[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def print_stats(
    name: str,
    values: List[float],
) -> None:

    print()
    print(name)
    print("-" * len(name))

    if not values:
        print("No values.")
        return

    arr = np.asarray(values, dtype=np.float64)

    print(f"N       : {len(arr)}")
    print(f"Mean    : {np.mean(arr):.6f}")
    print(f"Std     : {np.std(arr):.6f}")
    print(f"Median  : {np.median(arr):.6f}")
    print(f"Min     : {np.min(arr):.6f}")
    print(f"Max     : {np.max(arr):.6f}")

    print()
    print("Percentiles:")

    for p in PERCENTILES:
        print(
            f"  P{p:<2} : "
            f"{np.percentile(arr, p):.6f}"
        )


def classify_gap(gap: float) -> str:
    """
    Classify a REAL recurrence gap.

    gap=0 is reserved for the no-prior placeholder.
    Real recurrence gaps are strictly > 0 for a non-decreasing
    event timeline unless duplicate timestamps exist.
    """

    if gap <= 0.0:
        return "NO_PRIOR"

    if gap <= 6.0:
        return "0-6h"

    if gap <= 24.0:
        return "6-24h"

    if gap <= 72.0:
        return "24-72h"

    if gap <= 168.0:
        return "72-168h"

    if gap <= 336.0:
        return "168-336h"

    return ">336h"


def bucket_order() -> List[str]:
    return [
        "NO_PRIOR",
        "0-6h",
        "6-24h",
        "24-72h",
        "72-168h",
        "168-336h",
        ">336h",
    ]


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_recurrence_data(
    records: List[dict],
) -> Dict[str, object]:

    all_real_gaps: List[float] = []
    positive_real_gaps: List[float] = []
    negative_real_gaps: List[float] = []

    all_norm_gaps: List[float] = []
    positive_norm_gaps: List[float] = []
    negative_norm_gaps: List[float] = []

    labels: List[int] = []

    # One score per window:
    # recurrence density = fraction of events having a prior
    # same-category occurrence.
    recurrence_density: List[float] = []

    positive_recurrence_density: List[float] = []
    negative_recurrence_density: List[float] = []

    # Minimum recurrence gap in each window.
    min_recurrence_gap: List[float] = []
    positive_min_gap: List[float] = []
    negative_min_gap: List[float] = []

    # Mean recurrence gap per window, only over actual recurrence events.
    mean_recurrence_gap: List[float] = []
    positive_mean_gap: List[float] = []
    negative_mean_gap: List[float] = []

    # Bucket counts by window label.
    bucket_counts_positive = {
        bucket: 0 for bucket in bucket_order()
    }

    bucket_counts_negative = {
        bucket: 0 for bucket in bucket_order()
    }

    # Event-level counts.
    event_bucket_positive = {
        bucket: 0 for bucket in bucket_order()
    }

    event_bucket_negative = {
        bucket: 0 for bucket in bucket_order()
    }

    total_events_positive = 0
    total_events_negative = 0

    recurrence_events_positive = 0
    recurrence_events_negative = 0

    no_prior_events_positive = 0
    no_prior_events_negative = 0

    for record in records:

        label = int(record["label"])
        labels.append(label)

        events = record["events"]

        category_ids = [
            int(event["category_id"])
            for event in events
        ]

        raw_hours = [
            float(event["delta_hours"])
            for event in events
        ]

        # ------------------------------------------------------------
        # Compute recurrence exactly as dataset_v2.py does.
        # ------------------------------------------------------------

        last_seen: Dict[int, float] = {}

        window_gaps: List[float] = []
        window_norm_gaps: List[float] = []

        for i, category in enumerate(category_ids):

            current_time = raw_hours[i]

            if category in last_seen:

                gap = current_time - last_seen[category]

                # Defensive check.
                if gap < -1e-6:
                    raise ValueError(
                        "Negative recurrence gap encountered. "
                        "Input timestamps are not non-decreasing."
                    )

                gap = max(0.0, gap)

                window_gaps.append(gap)

                logged = math.log1p(gap)

                # Normalization is added later after we have
                # train statistics.
                window_norm_gaps.append(logged)

            last_seen[category] = current_time

        total_events = len(events)
        recurrence_count = len(window_gaps)

        density = (
            recurrence_count / total_events
            if total_events > 0
            else 0.0
        )

        if window_gaps:
            min_gap = min(window_gaps)
            mean_gap = float(np.mean(window_gaps))
        else:
            min_gap = float("inf")
            mean_gap = float("inf")

        recurrence_density.append(density)
        min_recurrence_gap.append(min_gap)
        mean_recurrence_gap.append(mean_gap)

        if label == 1:

            total_events_positive += total_events
            recurrence_events_positive += recurrence_count
            no_prior_events_positive += (
                total_events - recurrence_count
            )

            positive_recurrence_density.append(density)

            if window_gaps:
                positive_min_gap.append(min_gap)
                positive_mean_gap.append(mean_gap)

        else:

            total_events_negative += total_events
            recurrence_events_negative += recurrence_count
            no_prior_events_negative += (
                total_events - recurrence_count
            )

            negative_recurrence_density.append(density)

            if window_gaps:
                negative_min_gap.append(min_gap)
                negative_mean_gap.append(mean_gap)

        # ------------------------------------------------------------
        # Bucket recurrence events.
        # ------------------------------------------------------------

        if label == 1:

            # Every event with no prior same-category event belongs
            # to NO_PRIOR for event-level accounting.
            event_bucket_positive["NO_PRIOR"] += (
                total_events - recurrence_count
            )

            for gap in window_gaps:

                bucket = classify_gap(gap)

                event_bucket_positive[bucket] += 1

                all_real_gaps.append(gap)
                positive_real_gaps.append(gap)

        else:

            event_bucket_negative["NO_PRIOR"] += (
                total_events - recurrence_count
            )

            for gap in window_gaps:

                bucket = classify_gap(gap)

                event_bucket_negative[bucket] += 1

                all_real_gaps.append(gap)
                negative_real_gaps.append(gap)

        # ------------------------------------------------------------
        # Window-level bucket presence.
        #
        # A window is counted once per bucket if it contains at
        # least one recurrence event in that bucket.
        # ------------------------------------------------------------

        present_buckets = set(
            classify_gap(gap)
            for gap in window_gaps
        )

        if not window_gaps:
            present_buckets.add("NO_PRIOR")

        for bucket in present_buckets:

            if label == 1:
                bucket_counts_positive[bucket] += 1
            else:
                bucket_counts_negative[bucket] += 1

    return {
        "labels": labels,

        "all_real_gaps": all_real_gaps,
        "positive_real_gaps": positive_real_gaps,
        "negative_real_gaps": negative_real_gaps,

        "recurrence_density": recurrence_density,
        "positive_recurrence_density": positive_recurrence_density,
        "negative_recurrence_density": negative_recurrence_density,

        "min_recurrence_gap": min_recurrence_gap,
        "positive_min_gap": positive_min_gap,
        "negative_min_gap": negative_min_gap,

        "mean_recurrence_gap": mean_recurrence_gap,
        "positive_mean_gap": positive_mean_gap,
        "negative_mean_gap": negative_mean_gap,

        "bucket_counts_positive": bucket_counts_positive,
        "bucket_counts_negative": bucket_counts_negative,

        "event_bucket_positive": event_bucket_positive,
        "event_bucket_negative": event_bucket_negative,

        "total_events_positive": total_events_positive,
        "total_events_negative": total_events_negative,

        "recurrence_events_positive": recurrence_events_positive,
        "recurrence_events_negative": recurrence_events_negative,

        "no_prior_events_positive": no_prior_events_positive,
        "no_prior_events_negative": no_prior_events_negative,
    }


# ============================================================================
# NORMALIZED GAP ANALYSIS
# ============================================================================

def analyze_normalized_gap(
    records: List[dict],
    time_stats: Dict[str, float],
) -> None:

    mean = time_stats["gap_same_cat_mean"]
    std = time_stats["gap_same_cat_std"]

    positive_norm: List[float] = []
    negative_norm: List[float] = []

    for record in records:

        label = int(record["label"])

        events = record["events"]

        category_ids = [
            int(e["category_id"])
            for e in events
        ]

        raw_hours = [
            float(e["delta_hours"])
            for e in events
        ]

        last_seen: Dict[int, float] = {}

        for category, current_time in zip(
            category_ids,
            raw_hours,
        ):

            if category in last_seen:

                gap = current_time - last_seen[category]

                gap = max(0.0, gap)

                logged = math.log1p(gap)

                normalized = (
                    (logged - mean) / std
                    if std >= 1e-8
                    else logged - mean
                )

                if label == 1:
                    positive_norm.append(normalized)
                else:
                    negative_norm.append(normalized)

            last_seen[category] = current_time

    print()
    print("=" * 78)
    print("NORMALIZED RECURRENCE GAP")
    print("=" * 78)

    print_stats(
        "Positive windows — gap_same_category_norm",
        positive_norm,
    )

    print_stats(
        "Negative windows — gap_same_category_norm",
        negative_norm,
    )

    print()
    print(
        f"No-prior representation: "
        f"{math.log1p(0.0):.6f} -> "
        f"{((0.0 - mean) / std):.6f}"
    )


# ============================================================================
# BUCKET ANALYSIS
# ============================================================================

def print_bucket_analysis(
    data: Dict[str, object],
) -> None:

    positive_window_counts = data["bucket_counts_positive"]
    negative_window_counts = data["bucket_counts_negative"]

    positive_event_counts = data["event_bucket_positive"]
    negative_event_counts = data["event_bucket_negative"]

    positive_windows = 0
    negative_windows = 0

    labels = data["labels"]

    for label in labels:
        if label == 1:
            positive_windows += 1
        else:
            negative_windows += 1

    print()
    print("=" * 78)
    print("WINDOW-LEVEL RECURRENCE BUCKET PRESENCE")
    print("=" * 78)

    print()
    print(
        f"{'Bucket':<15}"
        f"{'Pos N':>10}"
        f"{'Pos %':>10}"
        f"{'Neg N':>10}"
        f"{'Neg %':>10}"
        f"{'Pos/Neg enrichment':>20}"
    )

    print("-" * 78)

    for bucket in bucket_order():

        pos_n = positive_window_counts[bucket]
        neg_n = negative_window_counts[bucket]

        pos_rate = (
            pos_n / positive_windows
            if positive_windows
            else 0.0
        )

        neg_rate = (
            neg_n / negative_windows
            if negative_windows
            else 0.0
        )

        enrichment = (
            pos_rate / neg_rate
            if neg_rate > 1e-12
            else float("inf")
        )

        enrichment_text = (
            f"{enrichment:.3f}"
            if math.isfinite(enrichment)
            else "inf"
        )

        print(
            f"{bucket:<15}"
            f"{pos_n:>10}"
            f"{pos_rate:>9.3%}"
            f"{neg_n:>10}"
            f"{neg_rate:>9.3%}"
            f"{enrichment_text:>20}"
        )

    print()
    print("=" * 78)
    print("EVENT-LEVEL RECURRENCE BUCKET DISTRIBUTION")
    print("=" * 78)

    print()
    print(
        f"{'Bucket':<15}"
        f"{'Positive N':>15}"
        f"{'Positive %':>15}"
        f"{'Negative N':>15}"
        f"{'Negative %':>15}"
    )

    print("-" * 78)

    total_pos = data["total_events_positive"]
    total_neg = data["total_events_negative"]

    for bucket in bucket_order():

        pos_n = positive_event_counts[bucket]
        neg_n = negative_event_counts[bucket]

        pos_rate = (
            pos_n / total_pos
            if total_pos
            else 0.0
        )

        neg_rate = (
            neg_n / total_neg
            if total_neg
            else 0.0
        )

        print(
            f"{bucket:<15}"
            f"{pos_n:>15}"
            f"{pos_rate:>14.3%}"
            f"{neg_n:>15}"
            f"{neg_rate:>14.3%}"
        )


# ============================================================================
# WINDOW-LEVEL RECURRENCE STATISTICS
# ============================================================================

def print_window_statistics(
    data: Dict[str, object],
) -> None:

    print()
    print("=" * 78)
    print("WINDOW-LEVEL RECURRENCE STATISTICS")
    print("=" * 78)

    positive_density = data["positive_recurrence_density"]
    negative_density = data["negative_recurrence_density"]

    positive_min = data["positive_min_gap"]
    negative_min = data["negative_min_gap"]

    positive_mean = data["positive_mean_gap"]
    negative_mean = data["negative_mean_gap"]

    print_stats(
        "Positive windows — recurrence density",
        positive_density,
    )

    print_stats(
        "Negative windows — recurrence density",
        negative_density,
    )

    print_stats(
        "Positive windows — minimum recurrence gap",
        positive_min,
    )

    print_stats(
        "Negative windows — minimum recurrence gap",
        negative_min,
    )

    print_stats(
        "Positive windows — mean recurrence gap",
        positive_mean,
    )

    print_stats(
        "Negative windows — mean recurrence gap",
        negative_mean,
    )


# ============================================================================
# RECURRENCE-ONLY WINDOW SCORES
# ============================================================================

def recurrence_only_auc(
    labels: List[int],
    values: List[float],
    higher_is_positive: bool = True,
) -> Tuple[float, float]:

    y = np.asarray(labels, dtype=np.int64)
    x = np.asarray(values, dtype=np.float64)

    valid = np.isfinite(x)

    y = y[valid]
    x = x[valid]

    if not higher_is_positive:
        x = -x

    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    roc = roc_auc_score(y, x)
    pr = average_precision_score(y, x)

    return float(roc), float(pr)


def print_recurrence_only_diagnostics(
    data: Dict[str, object],
) -> None:

    labels = data["labels"]

    density = data["recurrence_density"]

    # Replace inf min gaps with a large finite value.
    min_gap = [
        value if math.isfinite(value) else 1e6
        for value in data["min_recurrence_gap"]
    ]

    mean_gap = [
        value if math.isfinite(value) else 1e6
        for value in data["mean_recurrence_gap"]
    ]

    print()
    print("=" * 78)
    print("RECURRENCE-ONLY WINDOW DIAGNOSTICS")
    print("=" * 78)

    # More recurrence = positive score.
    roc, pr = recurrence_only_auc(
        labels,
        density,
        higher_is_positive=True,
    )

    print()
    print("Recurrence density -> positive score")
    print(f"  ROC-AUC : {roc:.6f}")
    print(f"  PR-AUC  : {pr:.6f}")

    # Smaller minimum gap = stronger recurrence score.
    roc, pr = recurrence_only_auc(
        labels,
        min_gap,
        higher_is_positive=False,
    )

    print()
    print("Minimum recurrence gap -> positive score")
    print(f"  ROC-AUC : {roc:.6f}")
    print(f"  PR-AUC  : {pr:.6f}")

    # Smaller mean gap = stronger recurrence score.
    roc, pr = recurrence_only_auc(
        labels,
        mean_gap,
        higher_is_positive=False,
    )

    print()
    print("Mean recurrence gap -> positive score")
    print(f"  ROC-AUC : {roc:.6f}")
    print(f"  PR-AUC  : {pr:.6f}")

    print()
    print(
        "Interpretation:"
    )
    print(
        "  These are NOT model performance metrics."
    )
    print(
        "  They measure how much label information is present "
        "in simple recurrence statistics alone."
    )


# ============================================================================
# SUGGEST BUCKET BOUNDARIES
# ============================================================================

def print_quantile_based_boundaries(
    data: Dict[str, object],
) -> None:

    gaps = np.asarray(
        data["all_real_gaps"],
        dtype=np.float64,
    )

    if len(gaps) == 0:
        return

    print()
    print("=" * 78)
    print("RECURRENCE GAP QUANTILE BOUNDARIES")
    print("=" * 78)

    print()
    print(
        "These are candidate data-driven boundaries, "
        "not final model design."
    )

    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:

        value = float(np.percentile(gaps, p))

        print(
            f"  P{p:02d}: {value:.3f} hours"
        )


# ============================================================================
# FINAL INTERPRETATION
# ============================================================================

def print_final_interpretation(
    data: Dict[str, object],
) -> None:

    positive_density = np.asarray(
        data["positive_recurrence_density"],
        dtype=np.float64,
    )

    negative_density = np.asarray(
        data["negative_recurrence_density"],
        dtype=np.float64,
    )

    positive_gaps = np.asarray(
        data["positive_real_gaps"],
        dtype=np.float64,
    )

    negative_gaps = np.asarray(
        data["negative_real_gaps"],
        dtype=np.float64,
    )

    print()
    print("=" * 78)
    print("DIAGNOSTIC INTERPRETATION")
    print("=" * 78)

    print()

    print(
        "1. RECURRENCE COVERAGE"
    )

    print(
        f"   Positive mean recurrence density : "
        f"{np.mean(positive_density):.4f}"
    )

    print(
        f"   Negative mean recurrence density : "
        f"{np.mean(negative_density):.4f}"
    )

    density_ratio = (
        np.mean(positive_density)
        / np.mean(negative_density)
        if np.mean(negative_density) > 1e-12
        else float("inf")
    )

    print(
        f"   Density ratio (positive/negative): "
        f"{density_ratio:.4f}"
    )

    print()

    print(
        "2. RECURRENCE GAP"
    )

    if len(positive_gaps) > 0:
        print(
            f"   Positive median gap : "
            f"{np.median(positive_gaps):.3f}h"
        )

    if len(negative_gaps) > 0:
        print(
            f"   Negative median gap : "
            f"{np.median(negative_gaps):.3f}h"
        )

    print()

    print(
        "3. WHAT TO LOOK FOR"
    )

    print(
        "   A) If positive and negative recurrence distributions "
        "are strongly separated:"
    )

    print(
        "      recurrence is a genuine dataset signal."
    )

    print(
        "      Do NOT remove recurrence."
    )

    print()

    print(
        "   B) If distributions overlap substantially but V3-A "
        "collapses when gap_same_category is removed:"
    )

    print(
        "      the continuous recurrence representation may be "
        "too dominant / brittle."
    )

    print(
        "      Bucketed or discretized recurrence becomes a strong "
        "next experiment."
    )

    print()

    print(
        "   C) If recurrence density itself has very high "
        "ROC-AUC:"
    )

    print(
        "      the dataset contains a very strong recurrence "
        "shortcut."
    )

    print(
        "      We should consider regularizing recurrence rather "
        "than eliminating it."
    )

    print()

    print(
        "   D) If recurrence-only AUC is modest but V3-A relies "
        "heavily on gap_same_category:"
    )

    print(
        "      this is stronger evidence for representation "
        "over-dependence rather than an inherently dominant "
        "dataset signal."
    )

    print()

    print(
        "NEXT STEP"
    )

    print(
        "   Do NOT change V3-A yet."
    )

    print(
        "   Use this report to choose the recurrence representation "
        "for the next controlled experiment."
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 78)
    print("SOCIA — V3-A RECURRENCE DISTRIBUTION DIAGNOSTIC")
    print("=" * 78)

    print()
    print(f"Data directory : {DATA_DIR}")

    # ----------------------------------------------------------------
    # Build exact V2 datasets.
    # ----------------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
        time_stats,
    ) = build_datasets_v2(
        data_dir=DATA_DIR,
    )

    print()
    print("=" * 78)
    print("DATASET")
    print("=" * 78)

    print(
        f"Train samples : {len(train_dataset)}"
    )

    print(
        f"Val samples   : {len(val_dataset)}"
    )

    print(
        f"Test samples  : {len(test_dataset)}"
    )

    print()
    print("Train-fitted time statistics:")

    for key, value in time_stats.items():

        print(
            f"  {key:<25}: {value:.6f}"
        )

    # ----------------------------------------------------------------
    # Extract raw records.
    # ----------------------------------------------------------------

    test_records = test_dataset.records

    positive_windows = sum(
        int(record["label"]) == 1
        for record in test_records
    )

    negative_windows = sum(
        int(record["label"]) == 0
        for record in test_records
    )

    print()
    print(
        f"Test positive windows : {positive_windows}"
    )

    print(
        f"Test negative windows : {negative_windows}"
    )

    # ----------------------------------------------------------------
    # Collect recurrence information.
    # ----------------------------------------------------------------

    data = collect_recurrence_data(
        test_records
    )

    # ----------------------------------------------------------------
    # Basic event counts.
    # ----------------------------------------------------------------

    print()
    print("=" * 78)
    print("EVENT-LEVEL RECURRENCE COVERAGE")
    print("=" * 78)

    pos_total = data["total_events_positive"]
    neg_total = data["total_events_negative"]

    pos_recur = data["recurrence_events_positive"]
    neg_recur = data["recurrence_events_negative"]

    pos_no_prior = data["no_prior_events_positive"]
    neg_no_prior = data["no_prior_events_negative"]

    print()
    print(
        "Positive windows:"
    )

    print(
        f"  Total events        : {pos_total}"
    )

    print(
        f"  Recurrence events   : {pos_recur}"
    )

    print(
        f"  No-prior events     : {pos_no_prior}"
    )

    print(
        f"  Recurrence rate     : "
        f"{pos_recur / pos_total:.4f}"
    )

    print()

    print(
        "Negative windows:"
    )

    print(
        f"  Total events        : {neg_total}"
    )

    print(
        f"  Recurrence events   : {neg_recur}"
    )

    print(
        f"  No-prior events     : {neg_no_prior}"
    )

    print(
        f"  Recurrence rate     : "
        f"{neg_recur / neg_total:.4f}"
    )

    # ----------------------------------------------------------------
    # Raw gap statistics.
    # ----------------------------------------------------------------

    print()
    print("=" * 78)
    print("RAW RECURRENCE GAP DISTRIBUTION")
    print("=" * 78)

    print_stats(
        "All recurrence gaps",
        data["all_real_gaps"],
    )

    print_stats(
        "Positive-window recurrence gaps",
        data["positive_real_gaps"],
    )

    print_stats(
        "Negative-window recurrence gaps",
        data["negative_real_gaps"],
    )

    # ----------------------------------------------------------------
    # Normalized gap statistics.
    # ----------------------------------------------------------------

    analyze_normalized_gap(
        test_records,
        time_stats,
    )

    # ----------------------------------------------------------------
    # Window statistics.
    # ----------------------------------------------------------------

    print_window_statistics(
        data
    )

    # ----------------------------------------------------------------
    # Bucket analysis.
    # ----------------------------------------------------------------

    print_bucket_analysis(
        data
    )

    # ----------------------------------------------------------------
    # Recurrence-only AUC.
    # ----------------------------------------------------------------

    print_recurrence_only_diagnostics(
        data
    )

    # ----------------------------------------------------------------
    # Quantile boundaries.
    # ----------------------------------------------------------------

    print_quantile_based_boundaries(
        data
    )

    # ----------------------------------------------------------------
    # Final interpretation.
    # ----------------------------------------------------------------

    print_final_interpretation(
        data
    )

    print()
    print("=" * 78)
    print("RECURRENCE DISTRIBUTION DIAGNOSTIC COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()