from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# =============================================================================
# MODEL VERSIONS
# =============================================================================

MODEL_VERSIONS = (
    "v2",
    "v3a",
    "v3c",
    "v3a_bucketed_recurrence",
)


# =============================================================================
# DEVICE
# =============================================================================

def resolve_device(
    device: Optional[str | torch.device] = None,
) -> torch.device:
    """
    Resolve the torch device used by evaluation.

    If device is explicitly supplied, use it.
    Otherwise prefer CUDA when available, then CPU.
    """

    if device is not None:
        if isinstance(device, torch.device):
            return device

        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(
    version: str,
    checkpoint_path: Path,
    device: Optional[str | torch.device] = None,
):
    """
    Load a trained neural model checkpoint.

    `checkpoint_path` may be either:
      - the exact .pt checkpoint file
      - the checkpoint directory

    Returns:
        model, checkpoint_metadata, resolved_device
    """

    if version not in MODEL_VERSIONS:
        raise ValueError(
            f"Unknown model version: {version}. "
            f"Expected one of: {MODEL_VERSIONS}"
        )

    # -------------------------------------------------------------------------
    # Actual model classes used by this project
    # -------------------------------------------------------------------------

    if version == "v2":
        from ..models.model_v2 import (
            SociaPatternTransformerV2 as ModelClass
        )

    elif version == "v3a":
        from ..models.model_v3a import (
            SociaPatternTransformerV3A as ModelClass
        )

    elif version == "v3c":
        from ..models.model_v3c import (
            SociaPatternTransformerV3C as ModelClass
        )

    elif version == "v3a_bucketed_recurrence":
        from ..models.model_v3a_bucketed_recurrence import (
            SociaPatternTransformerV3ABucketedRecurrence as ModelClass
        )

    else:
        raise AssertionError(
            f"Unhandled model version: {version}"
        )

    # -------------------------------------------------------------------------
    # Resolve device
    # -------------------------------------------------------------------------

    resolved_device = resolve_device(device)

    # -------------------------------------------------------------------------
    # Resolve checkpoint path
    # -------------------------------------------------------------------------

    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        checkpoint_path = (
            checkpoint_path / f"best_model_{version}.pt"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {version}:\n"
            f"{checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint path is not a file:\n"
            f"{checkpoint_path}"
        )

    # -------------------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=resolved_device,
    )

    # -------------------------------------------------------------------------
    # Extract state dict
    # -------------------------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Checkpoint itself is a state_dict.
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # -------------------------------------------------------------------------
    # Extract metadata
    # -------------------------------------------------------------------------

    metadata: Dict[str, Any] = {}

    if isinstance(checkpoint, dict):
        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key not in {
                "model_state_dict",
                "state_dict",
            }
        }

    # -------------------------------------------------------------------------
    # Construct model
    # -------------------------------------------------------------------------

    try:
        model = ModelClass()

    except TypeError as exc:
        raise RuntimeError(
            f"Could not construct model '{version}' "
            f"using {ModelClass.__name__}().\n\n"
            f"Constructor error:\n{exc}"
        ) from exc

    # -------------------------------------------------------------------------
    # Load weights
    # -------------------------------------------------------------------------

    try:
        model.load_state_dict(state_dict)

    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint weights for '{version}'.\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Model class: {ModelClass.__name__}\n\n"
            f"Original error:\n{exc}"
        ) from exc

    model.to(resolved_device)
    model.eval()

    return model, metadata, resolved_device


# =============================================================================
# RECORD -> TENSOR CONVERSION
# =============================================================================

def records_to_tensors(
    records: List[Dict[str, Any]],
    time_stats: Dict[str, float],
    device: Optional[str | torch.device] = None,
):
    """
    Convert raw records into the tensor inputs expected by V2/V3-A/V3-C.

    The actual dataset_v2.py does NOT expose a records_to_tensors()
    function. Instead, it exposes:

        compute_temporal_features_single()

    We use that exact implementation here so evaluation reproduces the
    same temporal feature calculation used by the V2/V3-A/V3-C models.

    Returns:
        tuple:
            category_ids
            temporal_features
            raw_delta_hours

    Labels are intentionally excluded because inference does not need them.
    """

    from ..models.dataset_v2 import (
        compute_temporal_features_single,
    )

    resolved_device = resolve_device(device)

    category_batches: List[torch.Tensor] = []
    temporal_batches: List[torch.Tensor] = []
    raw_delta_batches: List[torch.Tensor] = []

    for record_index, record in enumerate(records):

        if "events" not in record:
            raise ValueError(
                f"Record {record_index} is missing 'events'."
            )

        events = record["events"]

        category_ids = [
            int(event["category_id"])
            for event in events
        ]

        raw_delta_hours = [
            float(event["delta_hours"])
            for event in events
        ]

        if len(category_ids) != len(raw_delta_hours):
            raise ValueError(
                f"Record {record_index} has inconsistent event lengths."
            )

        temporal_features = compute_temporal_features_single(
            category_ids,
            raw_delta_hours,
            time_stats,
        )

        category_batches.append(
            torch.tensor(
                category_ids,
                dtype=torch.long,
            )
        )

        temporal_batches.append(
            temporal_features
        )

        raw_delta_batches.append(
            torch.tensor(
                raw_delta_hours,
                dtype=torch.float32,
            )
        )

    if not category_batches:
        raise ValueError(
            "Cannot convert an empty record list to tensors."
        )

    category_ids_tensor = torch.stack(
        category_batches
    ).to(resolved_device)

    temporal_features_tensor = torch.stack(
        temporal_batches
    ).to(resolved_device)

    raw_delta_hours_tensor = torch.stack(
        raw_delta_batches
    ).to(resolved_device)

    return (
        category_ids_tensor,
        temporal_features_tensor,
        raw_delta_hours_tensor,
    )


def records_to_bucketed_tensors(
    records: List[Dict[str, Any]],
    time_stats: Dict[str, float],
    device: Optional[str | torch.device] = None,
):
    """
    Convert raw records into the tensor inputs expected by
    v3a_bucketed_recurrence.

    The actual dataset_v3a_bucketed.py exposes:

        compute_reduced_temporal_features_and_buckets_single()

    Returns:
        tuple:
            category_ids
            temporal_features
            raw_delta_hours
            recurrence_bucket_ids

    Labels are intentionally excluded because inference does not need them.
    """

    from ..models.dataset_v3a_bucketed import (
        compute_reduced_temporal_features_and_buckets_single,
    )

    resolved_device = resolve_device(device)

    category_batches: List[torch.Tensor] = []
    temporal_batches: List[torch.Tensor] = []
    raw_delta_batches: List[torch.Tensor] = []
    bucket_batches: List[torch.Tensor] = []

    for record_index, record in enumerate(records):

        if "events" not in record:
            raise ValueError(
                f"Record {record_index} is missing 'events'."
            )

        events = record["events"]

        category_ids = [
            int(event["category_id"])
            for event in events
        ]

        raw_delta_hours = [
            float(event["delta_hours"])
            for event in events
        ]

        if len(category_ids) != len(raw_delta_hours):
            raise ValueError(
                f"Record {record_index} has inconsistent event lengths."
            )

        (
            temporal_features,
            recurrence_bucket_ids,
        ) = compute_reduced_temporal_features_and_buckets_single(
            category_ids,
            raw_delta_hours,
            time_stats,
        )

        category_batches.append(
            torch.tensor(
                category_ids,
                dtype=torch.long,
            )
        )

        temporal_batches.append(
            temporal_features
        )

        raw_delta_batches.append(
            torch.tensor(
                raw_delta_hours,
                dtype=torch.float32,
            )
        )

        bucket_batches.append(
            recurrence_bucket_ids
        )

    if not category_batches:
        raise ValueError(
            "Cannot convert an empty record list to tensors."
        )

    category_ids_tensor = torch.stack(
        category_batches
    ).to(resolved_device)

    temporal_features_tensor = torch.stack(
        temporal_batches
    ).to(resolved_device)

    raw_delta_hours_tensor = torch.stack(
        raw_delta_batches
    ).to(resolved_device)

    recurrence_bucket_ids_tensor = torch.stack(
        bucket_batches
    ).to(resolved_device)

    return (
        category_ids_tensor,
        temporal_features_tensor,
        raw_delta_hours_tensor,
        recurrence_bucket_ids_tensor,
    )


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(
    model,
    records: List[Dict[str, Any]],
    time_stats: Dict[str, float],
    device: Optional[str | torch.device],
    *,
    version: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run neural-model inference.

    Argument order intentionally matches the Step 2 calls:

        run_inference(
            model,
            records,
            time_stats,
            device,
            version=version,
        )

    Returns:
        1D numpy array of probabilities in exactly the same
        order as `records`.
    """

    if version not in MODEL_VERSIONS:
        raise ValueError(
            f"Unknown model version: {version}. "
            f"Expected one of: {MODEL_VERSIONS}"
        )

    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be > 0, got {batch_size}."
        )

    if len(records) == 0:
        return np.asarray(
            [],
            dtype=np.float32,
        )

    resolved_device = resolve_device(device)

    # -------------------------------------------------------------------------
    # Build model inputs using the exact feature implementation for each
    # model family.
    # -------------------------------------------------------------------------

    if version == "v3a_bucketed_recurrence":

        tensors = records_to_bucketed_tensors(
            records,
            time_stats=time_stats,
            device=resolved_device,
        )

    else:

        tensors = records_to_tensors(
            records,
            time_stats=time_stats,
            device=resolved_device,
        )

    # -------------------------------------------------------------------------
    # Batched inference
    # -------------------------------------------------------------------------

    probabilities_all: List[np.ndarray] = []

    with torch.no_grad():

        for start in range(
            0,
            len(records),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(records),
            )

            batch = _slice_batch(
                tensors,
                start,
                end,
            )

            output = _call_model(
                model,
                batch,
                version,
            )

            # Some implementations may return:
            #
            #     (logits, auxiliary_output)
            #
            # In that case the first element is the classifier output.
            if isinstance(output, (tuple, list)):

                if len(output) == 0:
                    raise RuntimeError(
                        f"Model '{version}' returned an empty "
                        f"tuple/list."
                    )

                output = output[0]

            if not torch.is_tensor(output):
                raise TypeError(
                    f"Model '{version}' returned "
                    f"{type(output)} instead of a torch.Tensor."
                )

            logits = output

            # Expected binary classifier output:
            #
            #     [B]
            #
            # or:
            #
            #     [B, 1]
            if logits.ndim == 2 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)

            if logits.ndim != 1:
                raise ValueError(
                    f"Unexpected model output shape for "
                    f"{version}: {tuple(logits.shape)}"
                )

            probabilities = torch.sigmoid(
                logits
            )

            probabilities_all.append(
                probabilities.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    probabilities = np.concatenate(
        probabilities_all,
        axis=0,
    )

    if len(probabilities) != len(records):
        raise RuntimeError(
            f"Inference output length mismatch for {version}: "
            f"{len(probabilities)} probabilities for "
            f"{len(records)} records."
        )

    return probabilities


def _slice_batch(
    tensors,
    start: int,
    end: int,
):
    """
    Slice a tuple/list/dict of tensors along the first dimension.
    """

    if torch.is_tensor(tensors):
        return tensors[start:end]

    if isinstance(tensors, tuple):
        return tuple(
            _slice_batch(
                item,
                start,
                end,
            )
            for item in tensors
        )

    if isinstance(tensors, list):
        return [
            _slice_batch(
                item,
                start,
                end,
            )
            for item in tensors
        ]

    if isinstance(tensors, dict):
        return {
            key: _slice_batch(
                value,
                start,
                end,
            )
            for key, value in tensors.items()
        }

    return tensors


def _call_model(
    model,
    batch,
    version: str,
):
    """
    Call the model using the actual positional input structure used by
    the corresponding dataset.

    V2 / V3-A / V3-C:
        category_ids,
        temporal_features,
        raw_delta_hours

    V3-A bucketed:
        category_ids,
        temporal_features,
        raw_delta_hours,
        recurrence_bucket_ids

    Labels are intentionally not passed to the model.
    """

    if version == "v3a_bucketed_recurrence":

        if not isinstance(batch, (tuple, list)):
            raise TypeError(
                "Bucketed model input must be a tuple/list."
            )

        if len(batch) != 4:
            raise ValueError(
                "Bucketed model expects 4 input tensors "
                "(category_ids, temporal_features, "
                "raw_delta_hours, recurrence_bucket_ids), "
                f"got {len(batch)}."
            )

        return model(
            batch[0],
            batch[1],
            batch[2],
            batch[3],
        )

    if not isinstance(batch, (tuple, list)):
        raise TypeError(
            f"Model input for {version} must be a tuple/list."
        )

    if len(batch) != 3:
        raise ValueError(
            f"Model '{version}' expects 3 input tensors "
            "(category_ids, temporal_features, raw_delta_hours), "
            f"got {len(batch)}."
        )

    return model(
        batch[0],
        batch[1],
        batch[2],
    )


# =============================================================================
# PREDICTION ALIGNMENT
# =============================================================================

def align_predictions(
    records: List[Dict[str, Any]],
    probabilities: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Attach probabilities to records while preserving record order.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(records) != len(probabilities):
        raise ValueError(
            "Prediction/record length mismatch: "
            f"{len(records)} records vs "
            f"{len(probabilities)} predictions."
        )

    aligned = []

    for record, probability in zip(
        records,
        probabilities,
    ):
        row = dict(record)
        row["prob"] = float(probability)
        aligned.append(row)

    return aligned


# =============================================================================
# JSONL
# =============================================================================

def load_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:
    """
    Load JSONL while preserving file order.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    records: List[Dict[str, Any]] = []

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
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number} of {path}: {exc}"
                ) from exc

    return records


def write_jsonl(
    path: Path,
    records: List[Dict[str, Any]],
) -> None:
    """
    Write records as JSONL.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# =============================================================================
# COUNTERFACTUAL DATA
# =============================================================================

def load_counterfactual(
    path: Path,
) -> List[Dict[str, Any]]:
    """
    Load and validate counterfactual evaluation records.

    Required fields:
        sample_id
        counterfactual_group_id
        role
        transform
        base_sample_id

    Original file order is preserved.
    """

    rows = load_jsonl(path)

    required_fields = {
        "sample_id",
        "counterfactual_group_id",
        "role",
        "transform",
        "base_sample_id",
    }

    seen_sample_ids = set()

    for index, row in enumerate(rows):

        missing = (
            required_fields
            - set(row.keys())
        )

        if missing:
            raise ValueError(
                f"Counterfactual row {index} "
                f"is missing fields: "
                f"{sorted(missing)}"
            )

        sample_id = row["sample_id"]

        if not sample_id:
            raise ValueError(
                f"Counterfactual row {index} "
                f"has an empty sample_id."
            )

        if sample_id in seen_sample_ids:
            raise ValueError(
                f"Duplicate counterfactual "
                f"sample_id: {sample_id}"
            )

        seen_sample_ids.add(
            sample_id
        )

        if row["role"] not in {
            "base",
            "transformed",
        }:
            raise ValueError(
                f"Invalid counterfactual role "
                f"{row['role']!r} for "
                f"sample_id={sample_id!r}. "
                f"Expected 'base' or 'transformed'."
            )

    return rows


# =============================================================================
# THRESHOLD UTILITIES
# =============================================================================

def get_thresholds(
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> np.ndarray:
    """
    Return the fixed threshold grid used by Step 2.
    """

    if step <= 0:
        raise ValueError(
            "Threshold step must be > 0."
        )

    if not 0 <= start <= 1:
        raise ValueError(
            "Threshold start must be in [0, 1]."
        )

    if not 0 <= stop <= 1:
        raise ValueError(
            "Threshold stop must be in [0, 1]."
        )

    if start > stop:
        raise ValueError(
            f"Threshold start ({start}) "
            f"cannot exceed stop ({stop})."
        )

    thresholds = np.arange(
        start,
        stop + step * 0.5,
        step,
        dtype=np.float64,
    )

    return np.round(
        thresholds,
        decimals=10,
    )


# =============================================================================
# METRICS
# =============================================================================

def _binary_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Return:
        TN, FP, FN, TP
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: "
            f"y_true={len(y_true)}, "
            f"y_pred={len(y_pred)}"
        )

    tn = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    return tn, fp, fn, tp


def compute_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute binary metrics at a fixed threshold.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(y_true) != len(probabilities):
        raise ValueError(
            f"Length mismatch: "
            f"y_true={len(y_true)}, "
            f"probabilities={len(probabilities)}"
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        _binary_confusion_counts(
            y_true,
            predictions,
        )
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / len(y_true)
        if len(y_true) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def sweep_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the threshold maximizing F1.

    Ties are resolved in favor of the lower threshold.
    """

    if thresholds is None:
        thresholds = get_thresholds()

    best_threshold = None
    best_metrics = None

    for threshold in thresholds:

        metrics = compute_binary_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        if (
            best_metrics is None
            or metrics["f1"] > best_metrics["f1"]
            or (
                metrics["f1"] == best_metrics["f1"]
                and float(threshold)
                < float(best_threshold)
            )
        ):
            best_threshold = float(
                threshold
            )

            best_metrics = metrics

    if (
        best_threshold is None
        or best_metrics is None
    ):
        raise RuntimeError(
            "Threshold sweep produced "
            "no candidate threshold."
        )

    return (
        best_threshold,
        best_metrics,
    )


def compute_full_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Compatibility wrapper for evaluation scripts.
    """

    return compute_binary_metrics(
        y_true=y_true,
        probabilities=probabilities,
        threshold=threshold,
    )


# =============================================================================
# JSON
# =============================================================================

def write_json(
    path: Path,
    data: Any,
) -> None:
    """
    Write JSON.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# =============================================================================
# EVALUATION METADATA
# =============================================================================

def load_eval_metadata_by_id(
    path: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Load evaluation metadata indexed by sample_id.
    """

    rows = load_jsonl(path)

    result: Dict[
        str,
        Dict[str, Any]
    ] = {}

    for row in rows:

        sample_id = row.get(
            "sample_id"
        )

        if sample_id is None:
            raise ValueError(
                "Evaluation metadata row "
                "is missing sample_id."
            )

        if sample_id in result:
            raise ValueError(
                "Duplicate sample_id in "
                f"evaluation metadata: "
                f"{sample_id}"
            )

        result[sample_id] = row

    return result