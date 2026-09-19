"""
Socia — Dataset V2

Adds two explicit relational temporal features on top of the existing
absolute-time signal, and returns raw (unnormalized) delta_hours
alongside the normalized features so the model can compute a
relative-time attention bias from real elapsed hours.

New per-event features (in addition to the existing normalized
absolute time):

    gap_prev          : hours since the immediately preceding event
                         in the window (0 for the first event).
    gap_same_category  : hours since the most recent PRIOR event of
                         the SAME category (this is the direct signal
                         for patterns like "A -> 2h -> B -> 3h -> A",
                         since it measures the A-to-A recurrence gap
                         specifically, not just adjacent-event gaps).
    has_prev_same_category : 1.0 if such a prior same-category event
                         exists in this window, else 0.0. Needed
                         because gap_same_category has no natural
                         value when there is no prior occurrence; we
                         use 0.0 as a placeholder and let the model
                         use this flag to know when to ignore it.

No changes to train/val/test.jsonl or the split are required — all of
this is derived from the existing 'category_id' / 'delta_hours'
fields already present in each record.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# Reuse loading/validation/dataloader logic from the V1 dataset module
# unchanged -- no reason to duplicate it, and it keeps V1 and V2
# guaranteed to agree on what counts as a valid record.
from dataset import (
    DEFAULT_DATA_DIR,
    EXPECTED_WINDOW_LENGTH,
    NUM_CATEGORIES,
    create_dataloader,
    load_jsonl,
    validate_dataset,
)

WINDOW_LENGTH = EXPECTED_WINDOW_LENGTH  # 25, kept for readability in V2 code

# temporal_features per event = [abs_time_norm, gap_prev_norm, gap_same_cat_norm, has_prev_same_cat]
TEMPORAL_FEATURE_DIM = 4


# =====================================================================
# TEMPORAL FEATURE ENGINEERING
# =====================================================================

def normalize_component(
    x: torch.Tensor,
    mean: float,
    std: float,
) -> torch.Tensor:
    """
    log1p + (x - mean) / std, matching the normalization convention
    already used for absolute delta_hours in V1's dataset.py.
    Assumes x >= 0 (true for all three raw quantities we normalize:
    absolute time, and both gap types, since gaps are computed as
    later_time - earlier_time on a non-decreasing timestamp sequence).
    """

    logged = torch.log1p(x)

    if std < 1e-8:
        return logged - mean

    return (logged - mean) / std


def compute_temporal_features_single(
    category_ids: List[int],
    raw_delta_hours: List[float],
    time_stats: Dict[str, float],
) -> torch.Tensor:
    """
    Build the (WINDOW_LENGTH, TEMPORAL_FEATURE_DIM) feature tensor for
    one window, given plain Python lists (not tensors) so this can be
    reused identically both in Dataset.__getitem__ (real windows) and
    in training-time augmentation (stretched / reordered windows).
    """

    n = len(category_ids)

    abs_hours = [float(x) for x in raw_delta_hours]

    # --- gap_prev: gap to the immediately preceding event ---
    gap_prev = [0.0] * n
    for i in range(1, n):
        gap_prev[i] = abs_hours[i] - abs_hours[i - 1]

    # --- gap_same_category / has_prev_same_category ---
    gap_same_cat = [0.0] * n
    has_prev_same_cat = [0.0] * n
    last_seen_time: Dict[int, float] = {}

    for i in range(n):
        cat = category_ids[i]
        if cat in last_seen_time:
            gap_same_cat[i] = abs_hours[i] - last_seen_time[cat]
            has_prev_same_cat[i] = 1.0
        # else: leave at 0.0 / 0.0 (flag off) -- no prior occurrence
        last_seen_time[cat] = abs_hours[i]

    abs_tensor = torch.tensor(abs_hours, dtype=torch.float32)
    gap_prev_tensor = torch.tensor(gap_prev, dtype=torch.float32)
    gap_same_cat_tensor = torch.tensor(gap_same_cat, dtype=torch.float32)
    has_prev_tensor = torch.tensor(has_prev_same_cat, dtype=torch.float32)

    abs_norm = normalize_component(
        abs_tensor, time_stats["abs_mean"], time_stats["abs_std"]
    )
    gap_prev_norm = normalize_component(
        gap_prev_tensor, time_stats["gap_prev_mean"], time_stats["gap_prev_std"]
    )
    gap_same_cat_norm = normalize_component(
        gap_same_cat_tensor,
        time_stats["gap_same_cat_mean"],
        time_stats["gap_same_cat_std"],
    )

    # Stack into (n, 4): [abs_norm, gap_prev_norm, gap_same_cat_norm, has_prev_flag]
    features = torch.stack(
        [abs_norm, gap_prev_norm, gap_same_cat_norm, has_prev_tensor],
        dim=-1,
    )

    return features


def fit_time_normalization_v2(records: List[dict]) -> Dict[str, float]:
    """
    Compute log1p mean/std separately for each of the three raw
    temporal quantities used in V2:

        - absolute delta_hours (same quantity V1 normalized)
        - gap_prev (excludes each window's first event, since its
          gap_prev is deterministically 0 and would bias the stats)
        - gap_same_category (only over occurrences that actually HAVE
          a prior same-category event -- the placeholder 0.0 values
          used when there's no history are excluded from these stats,
          since they aren't real gap measurements)
    """

    abs_logged: List[float] = []
    gap_prev_logged: List[float] = []
    gap_same_cat_logged: List[float] = []

    for record in records:
        events = record["events"]

        category_ids = [int(e["category_id"]) for e in events]
        raw_hours = [float(e["delta_hours"]) for e in events]

        n = len(events)
        last_seen: Dict[int, float] = {}

        for i in range(n):
            abs_logged.append(math.log1p(raw_hours[i]))

            if i > 0:
                gap = raw_hours[i] - raw_hours[i - 1]
                gap_prev_logged.append(math.log1p(gap))

            cat = category_ids[i]
            if cat in last_seen:
                same_cat_gap = raw_hours[i] - last_seen[cat]
                gap_same_cat_logged.append(math.log1p(same_cat_gap))

            last_seen[cat] = raw_hours[i]

    def stats_from(values: List[float]) -> Tuple[float, float]:
        tensor = torch.tensor(values, dtype=torch.float32)
        mean = tensor.mean().item()
        std = tensor.std(unbiased=False).item()

        if not math.isfinite(mean) or not math.isfinite(std):
            raise ValueError(
                "Computed a non-finite time normalization statistic."
            )

        if std < 1e-8:
            std = 1.0

        return mean, std

    abs_mean, abs_std = stats_from(abs_logged)

    # Defensive fallback in case a degenerate dataset has no non-first
    # events or no category repeats at all (shouldn't happen given
    # WINDOW_LENGTH=25 and NUM_CATEGORIES=15, but fail soft rather than
    # crash on an edge case that doesn't affect the modeling logic).
    gap_prev_mean, gap_prev_std = (
        stats_from(gap_prev_logged) if gap_prev_logged else (0.0, 1.0)
    )
    gap_same_cat_mean, gap_same_cat_std = (
        stats_from(gap_same_cat_logged) if gap_same_cat_logged else (0.0, 1.0)
    )

    return {
        "abs_mean": abs_mean,
        "abs_std": abs_std,
        "gap_prev_mean": gap_prev_mean,
        "gap_prev_std": gap_prev_std,
        "gap_same_cat_mean": gap_same_cat_mean,
        "gap_same_cat_std": gap_same_cat_std,
    }


# =====================================================================
# DATASET
# =====================================================================

class SociaWindowDatasetV2(Dataset):
    """
    Returns, per item:

        category_ids     : (WINDOW_LENGTH,) long
        temporal_features : (WINDOW_LENGTH, TEMPORAL_FEATURE_DIM) float32
                            [abs_norm, gap_prev_norm, gap_same_cat_norm, has_prev_flag]
        raw_delta_hours   : (WINDOW_LENGTH,) float32, UNNORMALIZED hours
                            (needed by the model's relative-time
                            attention bias, and by training-time
                            augmentation to build stretched/reordered
                            variants)
        label             : scalar float32
    """

    def __init__(
        self,
        records: List[dict],
        time_stats: Dict[str, float],
    ) -> None:

        self.records = records
        self.time_stats = time_stats

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        record = self.records[index]
        events = record["events"]

        category_ids_list = [int(e["category_id"]) for e in events]
        raw_hours_list = [float(e["delta_hours"]) for e in events]

        category_ids = torch.tensor(category_ids_list, dtype=torch.long)
        raw_delta_hours = torch.tensor(raw_hours_list, dtype=torch.float32)

        temporal_features = compute_temporal_features_single(
            category_ids_list, raw_hours_list, self.time_stats
        )

        label = torch.tensor(float(record["label"]), dtype=torch.float32)

        return category_ids, temporal_features, raw_delta_hours, label


def build_datasets_v2(
    data_dir: Optional[Path] = None,
) -> Tuple[
    SociaWindowDatasetV2,
    SociaWindowDatasetV2,
    SociaWindowDatasetV2,
    Dict[str, float],
]:
    """
    Same train/val/test split and same source files as V1 -- only the
    feature representation changes. Normalization statistics are fit
    on the TRAIN split only, exactly as in V1.
    """

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    data_dir = Path(data_dir)

    train_records = load_jsonl(data_dir / "train.jsonl")
    val_records = load_jsonl(data_dir / "val.jsonl")
    test_records = load_jsonl(data_dir / "test.jsonl")

    validate_dataset(train_records, "train")
    validate_dataset(val_records, "val")
    validate_dataset(test_records, "test")

    time_stats = fit_time_normalization_v2(train_records)

    train_dataset = SociaWindowDatasetV2(train_records, time_stats)
    val_dataset = SociaWindowDatasetV2(val_records, time_stats)
    test_dataset = SociaWindowDatasetV2(test_records, time_stats)

    return train_dataset, val_dataset, test_dataset, time_stats


def sanity_check_v2(data_dir: Optional[Path] = None, batch_size: int = 64) -> None:
    """Quick shape/dtype/range check, mirroring dataset.py's sanity_check()."""

    train_dataset, val_dataset, test_dataset, time_stats = build_datasets_v2(
        data_dir=data_dir
    )

    loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

    category_ids, temporal_features, raw_delta_hours, labels = next(iter(loader))

    expected_batch = min(batch_size, len(train_dataset))

    assert category_ids.shape == (expected_batch, WINDOW_LENGTH)
    assert temporal_features.shape == (expected_batch, WINDOW_LENGTH, TEMPORAL_FEATURE_DIM)
    assert raw_delta_hours.shape == (expected_batch, WINDOW_LENGTH)
    assert labels.shape == (expected_batch,)

    assert torch.isfinite(temporal_features).all()
    assert torch.isfinite(raw_delta_hours).all()
    assert torch.all((category_ids >= 0) & (category_ids < NUM_CATEGORIES))
    assert torch.all(raw_delta_hours >= 0)

    print("=" * 78)
    print("SOCIA DATASET V2 SANITY CHECK")
    print("=" * 78)
    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")
    print()
    print("Time normalization statistics (V2):")
    for key, value in time_stats.items():
        print(f"  {key}: {value:.6f}")
    print()
    print(f"category_ids shape      : {category_ids.shape}")
    print(f"temporal_features shape : {temporal_features.shape}")
    print(f"raw_delta_hours shape   : {raw_delta_hours.shape}")
    print()
    print("PASS")


if __name__ == "__main__":
    sanity_check_v2()