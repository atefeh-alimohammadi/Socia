from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset



EXPECTED_WINDOW_LENGTH = 25
NUM_CATEGORIES = 15

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "data"
)


def load_jsonl(path: Path) -> List[dict]:


    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{path}"
        )

    records: List[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {path}: {exc}"
                ) from exc

            records.append(record)

    return records



def validate_record(
    record: dict,
    index: int,
) -> None:


    if "events" not in record:
        raise ValueError(
            f"Record {index} is missing 'events'."
        )

    if "label" not in record:
        raise ValueError(
            f"Record {index} is missing 'label'."
        )

    events = record["events"]

    if not isinstance(events, list):
        raise TypeError(
            f"Record {index}: 'events' must be a list."
        )

    if len(events) != EXPECTED_WINDOW_LENGTH:
        raise ValueError(
            f"Record {index}: expected "
            f"{EXPECTED_WINDOW_LENGTH} events, "
            f"got {len(events)}."
        )

    label = record["label"]

    if label not in (0, 1):
        raise ValueError(
            f"Record {index}: label must be 0 or 1, "
            f"got {label!r}."
        )

    previous_delta = None

    for event_index, event in enumerate(events):

        if "category_id" not in event:
            raise ValueError(
                f"Record {index}, event {event_index}: "
                f"missing 'category_id'."
            )

        if "delta_hours" not in event:
            raise ValueError(
                f"Record {index}, event {event_index}: "
                f"missing 'delta_hours'."
            )

        category_id = event["category_id"]
        delta_hours = event["delta_hours"]

        if not isinstance(category_id, int):
            raise TypeError(
                f"Record {index}, event {event_index}: "
                f"category_id must be int."
            )

        if not 0 <= category_id < NUM_CATEGORIES:
            raise ValueError(
                f"Record {index}, event {event_index}: "
                f"category_id {category_id} outside "
                f"[0, {NUM_CATEGORIES - 1}]."
            )

        if not isinstance(
            delta_hours,
            (int, float),
        ):
            raise TypeError(
                f"Record {index}, event {event_index}: "
                f"delta_hours must be numeric."
            )

        if not math.isfinite(
            float(delta_hours)
        ):
            raise ValueError(
                f"Record {index}, event {event_index}: "
                f"delta_hours is not finite."
            )

        if delta_hours < 0:
            raise ValueError(
                f"Record {index}, event {event_index}: "
                f"delta_hours cannot be negative."
            )

        # delta_hours in the exported dataset is cumulative,
        # so it should be non-decreasing within a window.
        if (
            previous_delta is not None
            and delta_hours < previous_delta
        ):
            raise ValueError(
                f"Record {index}, event {event_index}: "
                f"delta_hours is not non-decreasing."
            )

        previous_delta = float(delta_hours)


def validate_dataset(
    records: List[dict],
    split_name: str,
) -> None:
    """
    Validate an entire split.
    """

    if not records:
        raise ValueError(
            f"{split_name} dataset is empty."
        )

    for index, record in enumerate(records):
        validate_record(
            record,
            index,
        )


def log1p_delta_hours(
    delta_hours: torch.Tensor,
) -> torch.Tensor:


    if torch.any(delta_hours < 0):
        raise ValueError(
            "delta_hours contains negative values."
        )

    return torch.log1p(
        delta_hours
    )


def fit_time_normalization(
    records: List[dict],
) -> Dict[str, float]:

    values: List[float] = []

    for record in records:

        for event in record["events"]:

            delta = float(
                event["delta_hours"]
            )

            values.append(
                math.log1p(delta)
            )

    tensor = torch.tensor(
        values,
        dtype=torch.float32,
    )

    mean = tensor.mean().item()
    std = tensor.std(
        unbiased=False
    ).item()

    if not math.isfinite(mean):
        raise ValueError(
            "Computed time normalization mean is not finite."
        )

    if not math.isfinite(std):
        raise ValueError(
            "Computed time normalization std is not finite."
        )

    if std < 1e-8:
        std = 1.0

    return {
        "mean": mean,
        "std": std,
    }


def normalize_delta_hours(
    delta_hours: torch.Tensor,
    time_stats: Dict[str, float],
) -> torch.Tensor:


    logged = log1p_delta_hours(
        delta_hours
    )

    mean = float(
        time_stats["mean"]
    )

    std = float(
        time_stats["std"]
    )

    normalized = (
        logged - mean
    ) / std

    return normalized


class SociaWindowDataset(Dataset):


    def __init__(
        self,
        records: List[dict],
        time_stats: Dict[str, float],
    ) -> None:

        self.records = records

        self.time_stats = time_stats

    def __len__(self) -> int:

        return len(
            self.records
        )

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        record = self.records[
            index
        ]

        events = record[
            "events"
        ]

        category_ids = torch.tensor(
            [
                event["category_id"]
                for event in events
            ],
            dtype=torch.long,
        )

        raw_delta_hours = torch.tensor(
            [
                float(event["delta_hours"])
                for event in events
            ],
            dtype=torch.float32,
        )

        delta_hours = normalize_delta_hours(
            raw_delta_hours,
            self.time_stats,
        )

        label = torch.tensor(
            float(record["label"]),
            dtype=torch.float32,
        )

        return (
            category_ids,
            delta_hours,
            label,
        )


def create_dataloader(
    dataset: SociaWindowDataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_datasets(
    data_dir: Optional[Path] = None,
) -> Tuple[
    SociaWindowDataset,
    SociaWindowDataset,
    SociaWindowDataset,
    Dict[str, float],
]:


    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    data_dir = Path(
        data_dir
    )

    train_path = (
        data_dir / "train.jsonl"
    )

    val_path = (
        data_dir / "val.jsonl"
    )

    test_path = (
        data_dir / "test.jsonl"
    )

    train_records = load_jsonl(
        train_path
    )

    val_records = load_jsonl(
        val_path
    )

    test_records = load_jsonl(
        test_path
    )

    validate_dataset(
        train_records,
        "train",
    )

    validate_dataset(
        val_records,
        "val",
    )

    validate_dataset(
        test_records,
        "test",
    )


    time_stats = fit_time_normalization(
        train_records
    )

    train_dataset = SociaWindowDataset(
        train_records,
        time_stats,
    )

    val_dataset = SociaWindowDataset(
        val_records,
        time_stats,
    )

    test_dataset = SociaWindowDataset(
        test_records,
        time_stats,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        time_stats,
    )


def sanity_check(
    data_dir: Optional[Path] = None,
    batch_size: int = 64,
) -> None:


    (
        train_dataset,
        val_dataset,
        test_dataset,
        time_stats,
    ) = build_datasets(
        data_dir=data_dir
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    (
        category_ids,
        delta_hours,
        labels,
    ) = next(
        iter(train_loader)
    )

    expected_batch = min(
        batch_size,
        len(train_dataset),
    )

    assert category_ids.shape == (
        expected_batch,
        EXPECTED_WINDOW_LENGTH,
    )

    assert delta_hours.shape == (
        expected_batch,
        EXPECTED_WINDOW_LENGTH,
    )

    assert labels.shape == (
        expected_batch,
    )

    assert category_ids.dtype == torch.long

    assert delta_hours.dtype == torch.float32

    assert labels.dtype == torch.float32

    assert torch.all(
        (category_ids >= 0)
        & (category_ids < NUM_CATEGORIES)
    )

    assert torch.all(
        (labels == 0)
        | (labels == 1)
    )

    category_finite = torch.isfinite(
        category_ids.float()
    ).all().item()

    delta_finite = torch.isfinite(
        delta_hours
    ).all().item()

    labels_finite = torch.isfinite(
        labels
    ).all().item()

    assert category_finite

    assert delta_finite

    assert labels_finite

    print()
    print("=" * 78)
    print("SOCIA DATASET LOADER SANITY CHECK")
    print("=" * 78)

    print()
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
    print("Train normalization statistics:")
    print(
        f"  log1p mean : {time_stats['mean']:.6f}"
    )
    print(
        f"  log1p std  : {time_stats['std']:.6f}"
    )

    print()
    print("Batch shapes:")
    print(
        f"  category_ids : {category_ids.shape}"
    )
    print(
        f"  delta_hours  : {delta_hours.shape}"
    )
    print(
        f"  labels       : {labels.shape}"
    )

    print()
    print("Dtypes:")
    print(
        f"  category_ids : {category_ids.dtype}"
    )
    print(
        f"  delta_hours  : {delta_hours.dtype}"
    )
    print(
        f"  labels       : {labels.dtype}"
    )

    print()
    print("Batch values:")
    print(
        f"  category_id min : "
        f"{category_ids.min().item()}"
    )
    print(
        f"  category_id max : "
        f"{category_ids.max().item()}"
    )
    print(
        f"  delta_hours min : "
        f"{delta_hours.min().item():.6f}"
    )
    print(
        f"  delta_hours max : "
        f"{delta_hours.max().item():.6f}"
    )

    print()
    print("Finite checks:")
    print(
        f"  category_ids NaN/Inf : "
        f"{not category_finite}"
    )
    print(
        f"  delta_hours NaN/Inf  : "
        f"{not delta_finite}"
    )
    print(
        f"  labels NaN/Inf       : "
        f"{not labels_finite}"
    )

    print()
    print("DataLoader creation:")
    print(
        f"  train : {len(train_loader)} batches"
    )
    print(
        f"  val   : {len(val_loader)} batches"
    )
    print(
        f"  test  : {len(test_loader)} batches"
    )

    print()
    print("=" * 78)
    print("STEP 1 SANITY CHECK: PASS")
    print("=" * 78)
    print()


if __name__ == "__main__":

    sanity_check()