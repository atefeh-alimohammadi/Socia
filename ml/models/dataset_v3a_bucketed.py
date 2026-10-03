"""
Socia -- Dataset V3-A Bucketed Recurrence (Experiment 1 support module)

Not one of the three requested deliverables by itself, but required by
all three of them: something has to turn raw events into the reduced
2-dim temporal_features + recurrence_bucket_ids tensors the new model
expects.

Rather than re-deriving abs_time_norm / gap_prev_norm from scratch
(risking silent drift from V2/V3-A's exact formula), this module
imports and reuses dataset_v2's normalization primitives directly:

    - dataset_v2.normalize_component        (exact same log1p + z-score)
    - dataset_v2.fit_time_normalization_v2  (exact same train-fit stats;
      gap_same_cat_mean/std are computed by this function too but are
      NOT used for bucketing here -- kept only so the returned
      time_stats dict is directly comparable/loggable against V2/V3-A's)

This guarantees abs_time_norm and gap_prev_norm in this experiment are
IDENTICAL, value for value, to what V3-A used -- only the recurrence
representation differs.

Recurrence bucket boundaries are FIXED / PREDEFINED (see
model_v3a_bucketed_recurrence.RECURRENCE_BUCKET_EDGES_HOURS), not fit
from data, and are the same for train/val/test. No test-label
information is used anywhere in this file.

Does not modify dataset.py or dataset_v2.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from dataset import (
    DEFAULT_DATA_DIR,
    EXPECTED_WINDOW_LENGTH,
    NUM_CATEGORIES,
    create_dataloader,
    load_jsonl,
    validate_dataset,
)

from dataset_v2 import (
    normalize_component,
    fit_time_normalization_v2,
)

from model_v3a_bucketed_recurrence import (
    NUM_RECURRENCE_BUCKETS,
    recurrence_bucket_id,
)

WINDOW_LENGTH = EXPECTED_WINDOW_LENGTH
REDUCED_TEMPORAL_FEATURE_DIM = 2


def compute_reduced_temporal_features_and_buckets_single(
    category_ids: List[int],
    raw_delta_hours: List[float],
    time_stats: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mirrors dataset_v2.compute_temporal_features_single's gap_prev /
    gap_same_category / has_prev_same_category computation exactly
    (same loop, same semantics, same raw values), but returns:

        temporal_features      : (n, 2) [abs_time_norm, gap_prev_norm]
        recurrence_bucket_ids  : (n,) long, in [0, NUM_RECURRENCE_BUCKETS)

    instead of V2/V3-A's 4-dim tensor.
    """

    n = len(category_ids)

    abs_hours = [float(x) for x in raw_delta_hours]

    gap_prev = [0.0] * n
    for i in range(1, n):
        gap_prev[i] = abs_hours[i] - abs_hours[i - 1]

    gap_same_cat = [0.0] * n
    has_prev_same_cat = [False] * n
    last_seen_time: Dict[int, float] = {}

    for i in range(n):
        cat = category_ids[i]
        if cat in last_seen_time:
            gap_same_cat[i] = abs_hours[i] - last_seen_time[cat]
            has_prev_same_cat[i] = True
        # else: leave at 0.0 / False -- no prior occurrence.
        last_seen_time[cat] = abs_hours[i]

    abs_tensor = torch.tensor(abs_hours, dtype=torch.float32)
    gap_prev_tensor = torch.tensor(gap_prev, dtype=torch.float32)

    abs_norm = normalize_component(
        abs_tensor, time_stats["abs_mean"], time_stats["abs_std"]
    )
    gap_prev_norm = normalize_component(
        gap_prev_tensor, time_stats["gap_prev_mean"], time_stats["gap_prev_std"]
    )

    temporal_features = torch.stack([abs_norm, gap_prev_norm], dim=-1)

    bucket_ids = torch.tensor(
        [
            recurrence_bucket_id(gap_same_cat[i], has_prev_same_cat[i])
            for i in range(n)
        ],
        dtype=torch.long,
    )

    return temporal_features, bucket_ids


class SociaWindowDatasetV3ABucketed(Dataset):
    """
    Returns, per item:

        category_ids           : (WINDOW_LENGTH,) long
        temporal_features        : (WINDOW_LENGTH, 2) float32
        raw_delta_hours          : (WINDOW_LENGTH,) float32
        recurrence_bucket_ids    : (WINDOW_LENGTH,) long
        label                    : scalar float32

    Same record source, same fields, same split as V2/V3-A -- only the
    recurrence representation differs.
    """

    def __init__(self, records: List[dict], time_stats: Dict[str, float]) -> None:
        self.records = records
        self.time_stats = time_stats

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):

        record = self.records[index]
        events = record["events"]

        category_ids_list = [int(e["category_id"]) for e in events]
        raw_hours_list = [float(e["delta_hours"]) for e in events]

        category_ids = torch.tensor(category_ids_list, dtype=torch.long)
        raw_delta_hours = torch.tensor(raw_hours_list, dtype=torch.float32)

        temporal_features, recurrence_bucket_ids = (
            compute_reduced_temporal_features_and_buckets_single(
                category_ids_list, raw_hours_list, self.time_stats
            )
        )

        label = torch.tensor(float(record["label"]), dtype=torch.float32)

        return category_ids, temporal_features, raw_delta_hours, recurrence_bucket_ids, label


def build_datasets_v3a_bucketed(
    data_dir: Optional[Path] = None,
) -> Tuple[
    SociaWindowDatasetV3ABucketed,
    SociaWindowDatasetV3ABucketed,
    SociaWindowDatasetV3ABucketed,
    Dict[str, float],
]:
    """
    Identical split/loading logic to build_datasets_v2 -- same
    train.jsonl / val.jsonl / test.jsonl, same validate_dataset call,
    same train-only stat fitting (via dataset_v2.fit_time_normalization_v2,
    reused directly, not reimplemented).
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

    # Reused unchanged from V2 -- fits abs/gap_prev/gap_same_cat stats
    # on TRAIN only. gap_same_cat_mean/std are computed but unused by
    # this experiment's model; kept in the dict only for logging
    # parity with V2/V3-A reports.
    time_stats = fit_time_normalization_v2(train_records)

    train_dataset = SociaWindowDatasetV3ABucketed(train_records, time_stats)
    val_dataset = SociaWindowDatasetV3ABucketed(val_records, time_stats)
    test_dataset = SociaWindowDatasetV3ABucketed(test_records, time_stats)

    return train_dataset, val_dataset, test_dataset, time_stats


def sanity_check_v3a_bucketed(data_dir: Optional[Path] = None, batch_size: int = 64) -> None:

    train_dataset, val_dataset, test_dataset, time_stats = build_datasets_v3a_bucketed(
        data_dir=data_dir
    )

    loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

    category_ids, temporal_features, raw_delta_hours, bucket_ids, labels = next(iter(loader))

    expected_batch = min(batch_size, len(train_dataset))

    assert category_ids.shape == (expected_batch, WINDOW_LENGTH)
    assert temporal_features.shape == (expected_batch, WINDOW_LENGTH, REDUCED_TEMPORAL_FEATURE_DIM)
    assert raw_delta_hours.shape == (expected_batch, WINDOW_LENGTH)
    assert bucket_ids.shape == (expected_batch, WINDOW_LENGTH)
    assert labels.shape == (expected_batch,)

    assert torch.isfinite(temporal_features).all()
    assert torch.isfinite(raw_delta_hours).all()
    assert torch.all((category_ids >= 0) & (category_ids < NUM_CATEGORIES))
    assert torch.all((bucket_ids >= 0) & (bucket_ids < NUM_RECURRENCE_BUCKETS))
    assert torch.all(raw_delta_hours >= 0)

    print("=" * 78)
    print("SOCIA DATASET V3-A BUCKETED -- SANITY CHECK")
    print("=" * 78)
    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")
    print()
    print(f"category_ids shape        : {category_ids.shape}")
    print(f"temporal_features shape   : {temporal_features.shape}")
    print(f"raw_delta_hours shape     : {raw_delta_hours.shape}")
    print(f"recurrence_bucket_ids     : {bucket_ids.shape}")
    print()
    print("Bucket id distribution (this training batch):")
    for b in range(NUM_RECURRENCE_BUCKETS):
        count = int((bucket_ids == b).sum().item())
        print(f"  bucket {b}: {count}")
    print()
    print("PASS")


if __name__ == "__main__":
    sanity_check_v3a_bucketed()
