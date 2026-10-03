from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# PATHS / CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent

# If this script is inside:
#
#   ml/eval/eval_recurrence_ablation_all.py
#
# then ml_root becomes:
#
#   ml/
#
ML_ROOT = ROOT.parent

CHECKPOINT_DIR = ML_ROOT / "models" / "checkpoints"


# ---------------------------------------------------------------------------
# Checkpoint names
#
# Change these only if your actual checkpoint filenames differ.
# ---------------------------------------------------------------------------

CHECKPOINTS = {
    "v1": CHECKPOINT_DIR / "best_model.pt",
    "v2": CHECKPOINT_DIR / "best_model_v2.pt",
    "v3a": CHECKPOINT_DIR / "best_model_v3a.pt",
    "v3c": CHECKPOINT_DIR / "best_model_v3c.pt",
    "v3a_bucketed": (
        CHECKPOINT_DIR / "best_model_v3a_bucketed_recurrence.pt"
    ),
}


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

GLOBAL_SEED = 20260829

# Option B uses a fixed permutation so that the result is reproducible.
OPTION_B_SEED = 20260829

# Fine-grained threshold diagnostic.
THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.99
THRESHOLD_STEP = 0.001


# =============================================================================
# MODEL IMPORTS
# =============================================================================

from model import SociaPatternTransformer
from model_v2 import SociaPatternTransformerV2
from model_v3a import SociaPatternTransformerV3A
from model_v3c import SociaPatternTransformerV3C

from model_v3a_bucketed_recurrence import (
    SociaPatternTransformerV3ABucketedRecurrence,
)


# =============================================================================
# DATASET IMPORTS
# =============================================================================

# V1
from dataset import build_datasets as build_datasets_v1

# V2 / V3-A / V3-C
from dataset_v2 import build_datasets_v2

# V3-A Bucketed
from dataset_v3a_bucketed import (
    build_datasets_v3a_bucketed,
)


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_CLASSES = {
    "v1": SociaPatternTransformer,
    "v2": SociaPatternTransformerV2,
    "v3a": SociaPatternTransformerV3A,
    "v3c": SociaPatternTransformerV3C,
    "v3a_bucketed": SociaPatternTransformerV3ABucketedRecurrence,
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Common checkpoint layouts.
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Some training scripts save the state_dict directly.
            state_dict = checkpoint

    else:
        raise TypeError(
            f"Unsupported checkpoint format: "
            f"{type(checkpoint)}"
        )

    # Handle DataParallel checkpoints.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    return (
        checkpoint
        if isinstance(checkpoint, dict)
        else {}
    )


def get_checkpoint_threshold(
    checkpoint: Dict[str, Any],
) -> float:

    possible_keys = [
        "val_threshold",
        "best_threshold",
        "threshold",
    ]

    for key in possible_keys:

        if key in checkpoint:

            value = float(
                checkpoint[key]
            )

            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"Checkpoint threshold '{key}' "
                    f"must be in (0, 1), got {value}."
                )

            return value

    raise KeyError(
        "No validation-derived threshold found in checkpoint. "
        "Refusing to silently invent an evaluation threshold."
    )


def get_checkpoint_epoch(
    checkpoint: Dict[str, Any],
) -> Optional[int]:

    for key in (
        "epoch",
        "best_epoch",
    ):

        if key in checkpoint:
            try:
                return int(
                    checkpoint[key]
                )
            except Exception:
                pass

    return None


def get_checkpoint_val_f1(
    checkpoint: Dict[str, Any],
) -> Optional[float]:

    for key in (
        "val_f1",
        "best_val_f1",
    ):

        if key in checkpoint:

            try:
                return float(
                    checkpoint[key]
                )
            except Exception:
                pass

    return None


# =============================================================================
# MODEL CONSTRUCTION
# =============================================================================

def create_model(
    model_name: str,
) -> torch.nn.Module:

    if model_name not in MODEL_CLASSES:
        raise KeyError(
            f"Unknown model: {model_name}"
        )

    model_class = MODEL_CLASSES[
        model_name
    ]

    model = model_class()

    return model.to(
        DEVICE
    )


# =============================================================================
# GENERIC METRICS
# =============================================================================

def safe_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            y_prob,
        )
    )


def safe_pr_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(
        average_precision_score(
            y_true,
            y_prob,
        )
    )


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:

    y_pred = (
        y_prob >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": safe_roc_auc(
            y_true,
            y_prob,
        ),
        "pr_auc": safe_pr_auc(
            y_true,
            y_prob,
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =============================================================================
# FINE-GRAINED THRESHOLD DIAGNOSTIC
# =============================================================================

def find_best_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step: float = THRESHOLD_STEP,
) -> Dict[str, float]:

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX + step / 2,
        step,
    )

    best_threshold = None
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for threshold in thresholds:

        y_pred = (
            y_prob >= threshold
        ).astype(np.int64)

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        if f1 > best_f1:

            best_f1 = float(f1)
            best_threshold = float(
                threshold
            )

            best_precision = float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            )

            best_recall = float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            )

    return {
        "threshold": best_threshold,
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
    }


# =============================================================================
# V1 DATA
# =============================================================================

def load_v1_test_data():
    (
        _train_dataset,
        _val_dataset,
        test_dataset,
        time_stats,
    ) = build_datasets_v1()

    return (
        test_dataset,
        time_stats,
    )


# =============================================================================
# V2 DATA
# =============================================================================

def load_v2_test_data():
    (
        _train_dataset,
        _val_dataset,
        test_dataset,
        time_stats,
    ) = build_datasets_v2()

    return (
        test_dataset,
        time_stats,
    )


# =============================================================================
# V3-A BUCKETED DATA
# =============================================================================

def load_bucketed_test_data():
    (
        _train_dataset,
        _val_dataset,
        test_dataset,
        time_stats,
    ) = build_datasets_v3a_bucketed()

    return (
        test_dataset,
        time_stats,
    )


# =============================================================================
# DATASET ITEM HELPERS
# =============================================================================

def unpack_v1_item(
    item: Any,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    if len(item) != 3:
        raise ValueError(
            "V1 dataset item must contain "
            "(category_ids, delta_hours, label). "
            f"Got {len(item)} fields."
        )

    category_ids = item[0]
    delta_hours = item[1]
    label = item[2]

    return (
        category_ids,
        delta_hours,
        label,
    )


def unpack_v2_item(
    item: Any,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    if len(item) != 4:
        raise ValueError(
            "V2/V3 dataset item must contain "
            "(category_ids, temporal_features, "
            "raw_delta_hours, label). "
            f"Got {len(item)} fields."
        )

    category_ids = item[0]
    temporal_features = item[1]
    raw_delta_hours = item[2]
    label = item[3]

    return (
        category_ids,
        temporal_features,
        raw_delta_hours,
        label,
    )


def unpack_bucketed_item(
    item: Any,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    if len(item) != 5:
        raise ValueError(
            "Bucketed dataset item must contain "
            "(category_ids, temporal_features, "
            "raw_delta_hours, recurrence_bucket_ids, label). "
            f"Got {len(item)} fields."
        )

    category_ids = item[0]
    temporal_features = item[1]
    raw_delta_hours = item[2]
    recurrence_bucket_ids = item[3]
    label = item[4]

    return (
        category_ids,
        temporal_features,
        raw_delta_hours,
        recurrence_bucket_ids,
        label,
    )


# =============================================================================
# V1 INFERENCE
# =============================================================================

@torch.no_grad()
def run_v1_inference(
    model: torch.nn.Module,
    dataset,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:

    model.eval()

    probabilities: List[float] = []
    labels: List[float] = []

    for index in range(
        len(dataset)
    ):

        (
            category_ids,
            delta_hours,
            label,
        ) = unpack_v1_item(
            dataset[index]
        )

        category_ids = (
            category_ids
            .unsqueeze(0)
            .to(DEVICE)
        )

        delta_hours = (
            delta_hours
            .unsqueeze(0)
            .to(DEVICE)
        )

        logits = model(
            category_ids,
            delta_hours,
        )

        probability = torch.sigmoid(
            logits
        ).reshape(-1)[0].item()

        probabilities.append(
            float(probability)
        )

        labels.append(
            float(label.item())
        )

    return (
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
    )


# =============================================================================
# V2 / V3-A / V3-C INFERENCE
# =============================================================================

@torch.no_grad()
def run_standard_inference(
    model: torch.nn.Module,
    dataset,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    model.eval()

    probabilities: List[float] = []
    labels: List[float] = []

    all_category_ids = []
    all_temporal_features = []
    all_raw_delta_hours = []

    for index in range(
        len(dataset)
    ):

        (
            category_ids,
            temporal_features,
            raw_delta_hours,
            label,
        ) = unpack_v2_item(
            dataset[index]
        )

        category_ids_batch = (
            category_ids
            .unsqueeze(0)
            .to(DEVICE)
        )

        temporal_features_batch = (
            temporal_features
            .unsqueeze(0)
            .to(DEVICE)
        )

        raw_delta_hours_batch = (
            raw_delta_hours
            .unsqueeze(0)
            .to(DEVICE)
        )

        logits = model(
            category_ids_batch,
            temporal_features_batch,
            raw_delta_hours_batch,
        )

        probability = torch.sigmoid(
            logits
        ).reshape(-1)[0].item()

        probabilities.append(
            float(probability)
        )

        labels.append(
            float(label.item())
        )

        all_category_ids.append(
            category_ids
        )

        all_temporal_features.append(
            temporal_features
        )

        all_raw_delta_hours.append(
            raw_delta_hours
        )

    return (
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
        torch.stack(
            all_category_ids
        ),
        torch.stack(
            all_temporal_features
        ),
        torch.stack(
            all_raw_delta_hours
        ),
    )


# =============================================================================
# BUCKETED INFERENCE
# =============================================================================

@torch.no_grad()
def run_bucketed_inference(
    model: torch.nn.Module,
    dataset,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    model.eval()

    probabilities: List[float] = []
    labels: List[float] = []

    all_category_ids = []
    all_temporal_features = []
    all_raw_delta_hours = []
    all_bucket_ids = []

    for index in range(
        len(dataset)
    ):

        (
            category_ids,
            temporal_features,
            raw_delta_hours,
            recurrence_bucket_ids,
            label,
        ) = unpack_bucketed_item(
            dataset[index]
        )

        category_ids_batch = (
            category_ids
            .unsqueeze(0)
            .to(DEVICE)
        )

        temporal_features_batch = (
            temporal_features
            .unsqueeze(0)
            .to(DEVICE)
        )

        raw_delta_hours_batch = (
            raw_delta_hours
            .unsqueeze(0)
            .to(DEVICE)
        )

        bucket_ids_batch = (
            recurrence_bucket_ids
            .unsqueeze(0)
            .to(DEVICE)
        )

        logits = model(
            category_ids_batch,
            temporal_features_batch,
            raw_delta_hours_batch,
            bucket_ids_batch,
        )

        probability = torch.sigmoid(
            logits
        ).reshape(-1)[0].item()

        probabilities.append(
            float(probability)
        )

        labels.append(
            float(label.item())
        )

        all_category_ids.append(
            category_ids
        )

        all_temporal_features.append(
            temporal_features
        )

        all_raw_delta_hours.append(
            raw_delta_hours
        )

        all_bucket_ids.append(
            recurrence_bucket_ids
        )

    return (
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
        torch.stack(
            all_category_ids
        ),
        torch.stack(
            all_temporal_features
        ),
        torch.stack(
            all_raw_delta_hours
        ),
        torch.stack(
            all_bucket_ids
        ),
    )


# =============================================================================
# TEMPORAL STATISTICS
# =============================================================================

def extract_stat(
    stats: Dict[str, Any],
    candidate_keys: Iterable[str],
) -> float:

    for key in candidate_keys:

        if key in stats:
            value = stats[key]

            if isinstance(
                value,
                dict,
            ):

                if "mean" in value:
                    return float(
                        value["mean"]
                    )

                if "std" in value:
                    return float(
                        value["std"]
                    )

            return float(value)

    raise KeyError(
        "Could not find any of the expected "
        f"normalization keys: {list(candidate_keys)}"
    )


def get_gap_same_cat_stats(
    time_stats: Dict[str, Any],
) -> Tuple[float, float]:

    mean_keys = [
        "gap_same_cat_mean",
        "same_category_gap_mean",
        "gap_same_category_mean",
    ]

    std_keys = [
        "gap_same_cat_std",
        "same_category_gap_std",
        "gap_same_category_std",
    ]

    mean = extract_stat(
        time_stats,
        mean_keys,
    )

    std = extract_stat(
        time_stats,
        std_keys,
    )

    if not math.isfinite(mean):
        raise ValueError(
            f"gap_same_category mean is not finite: {mean}"
        )

    if not math.isfinite(std):
        raise ValueError(
            f"gap_same_category std is not finite: {std}"
        )

    if std <= 0:
        raise ValueError(
            f"gap_same_category std must be > 0, got {std}"
        )

    return (
        mean,
        std,
    )


# =============================================================================
# OPTION A
#
# Remove recurrence information by replacing:
#
#   gap_same_category_norm
#   has_prev_same_category
#
# with the train-derived NO_PRIOR representation.
# =============================================================================

def make_option_a_temporal_features(
    temporal_features: torch.Tensor,
    time_stats: Dict[str, Any],
) -> torch.Tensor:

    if temporal_features.ndim != 3:
        raise ValueError(
            "Expected temporal_features shape "
            "[N, T, F]. "
            f"Got {tuple(temporal_features.shape)}"
        )

    if temporal_features.shape[-1] < 4:
        raise ValueError(
            "Option A requires the V2/V3 4D temporal "
            f"representation. Got F={temporal_features.shape[-1]}"
        )

    gap_same_cat_mean, gap_same_cat_std = (
        get_gap_same_cat_stats(
            time_stats
        )
    )

    output = (
        temporal_features
        .clone()
    )

    # Temporal feature layout:
    #
    # 0 = abs_time_norm
    # 1 = gap_prev_norm
    # 2 = gap_same_category_norm
    # 3 = has_prev_same_category

    no_prior_raw_gap = 0.0

    no_prior_normalized = (
        no_prior_raw_gap
        - gap_same_cat_mean
    ) / gap_same_cat_std

    output[:, :, 2] = (
        no_prior_normalized
    )

    output[:, :, 3] = 0.0

    return output


# =============================================================================
# OPTION B
#
# Globally permute recurrence-information pairs:
#
#   (gap_same_category_norm, has_prev_same_category)
#
# across all test events.
#
# This destroys event-local recurrence alignment while preserving
# the marginal distribution of the recurrence channels.
# =============================================================================

def make_option_b_temporal_features(
    temporal_features: torch.Tensor,
    seed: int = OPTION_B_SEED,
) -> torch.Tensor:

    if temporal_features.ndim != 3:
        raise ValueError(
            "Expected temporal_features shape "
            "[N, T, F]. "
            f"Got {tuple(temporal_features.shape)}"
        )

    if temporal_features.shape[-1] < 4:
        raise ValueError(
            "Option B requires the V2/V3 4D temporal "
            f"representation. Got F={temporal_features.shape[-1]}"
        )

    output = (
        temporal_features
        .clone()
    )

    n, t, _ = output.shape

    recurrence_pairs = output[
        :, :, 2:4
    ].reshape(
        n * t,
        2,
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed
    )

    permutation = torch.randperm(
        n * t,
        generator=generator,
    )

    permutation = permutation.to(
        recurrence_pairs.device
    )

    shuffled_pairs = (
        recurrence_pairs[
            permutation
        ]
    )

    output[:, :, 2:4] = (
        shuffled_pairs.reshape(
            n,
            t,
            2,
        )
    )

    return output


# =============================================================================
# BUCKETED OPTION B
#
# Shuffle recurrence bucket identities globally while leaving the
# non-recurrence temporal features unchanged.
# =============================================================================

def make_bucketed_option_b(
    bucket_ids: torch.Tensor,
    seed: int = OPTION_B_SEED,
) -> torch.Tensor:

    output = (
        bucket_ids
        .clone()
    )

    flat = output.reshape(
        -1
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed
    )

    permutation = torch.randperm(
        flat.numel(),
        generator=generator,
    )

    permutation = permutation.to(
        flat.device
    )

    shuffled = flat[
        permutation
    ]

    return shuffled.reshape(
        output.shape
    )


# =============================================================================
# STANDARD V2 / V3-A / V3-C ABLATION INFERENCE
# =============================================================================

@torch.no_grad()
def run_standard_with_modified_features(
    model: torch.nn.Module,
    category_ids: torch.Tensor,
    temporal_features: torch.Tensor,
    raw_delta_hours: torch.Tensor,
) -> np.ndarray:

    model.eval()

    probabilities = []

    n = category_ids.shape[0]

    for start in range(
        0,
        n,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            n,
        )

        cat = category_ids[
            start:end
        ].to(DEVICE)

        temp = temporal_features[
            start:end
        ].to(DEVICE)

        raw = raw_delta_hours[
            start:end
        ].to(DEVICE)

        logits = model(
            cat,
            temp,
            raw,
        )

        probs = torch.sigmoid(
            logits
        ).reshape(-1)

        probabilities.extend(
            probs.detach()
            .cpu()
            .tolist()
        )

    return np.asarray(
        probabilities,
        dtype=np.float64,
    )


# =============================================================================
# BUCKETED ABLATION INFERENCE
# =============================================================================

@torch.no_grad()
def run_bucketed_with_modified_features(
    model: torch.nn.Module,
    category_ids: torch.Tensor,
    temporal_features: torch.Tensor,
    raw_delta_hours: torch.Tensor,
    bucket_ids: torch.Tensor,
) -> np.ndarray:

    model.eval()

    probabilities = []

    n = category_ids.shape[0]

    for start in range(
        0,
        n,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            n,
        )

        cat = category_ids[
            start:end
        ].to(DEVICE)

        temp = temporal_features[
            start:end
        ].to(DEVICE)

        raw = raw_delta_hours[
            start:end
        ].to(DEVICE)

        buckets = bucket_ids[
            start:end
        ].to(DEVICE)

        logits = model(
            cat,
            temp,
            raw,
            buckets,
        )

        probs = torch.sigmoid(
            logits
        ).reshape(-1)

        probabilities.extend(
            probs.detach()
            .cpu()
            .tolist()
        )

    return np.asarray(
        probabilities,
        dtype=np.float64,
    )


# =============================================================================
# DELTA ANALYSIS
# =============================================================================

def probability_delta_summary(
    original_probabilities: np.ndarray,
    modified_probabilities: np.ndarray,
) -> Dict[str, float]:

    delta = (
        modified_probabilities
        - original_probabilities
    )

    return {
        "mean_delta": float(
            np.mean(delta)
        ),
        "median_delta": float(
            np.median(delta)
        ),
        "std_delta": float(
            np.std(delta)
        ),
        "mean_abs_delta": float(
            np.mean(
                np.abs(delta)
            )
        ),
        "expected_direction_rate": float(
            np.mean(
                delta < 0
            )
        ),
    }


# =============================================================================
# SAVE JSON
# =============================================================================

def save_json(
    path: Path,
    data: Dict[str, Any],
) -> None:

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
# PRINT METRICS
# =============================================================================

def print_metrics(
    name: str,
    metrics: Dict[str, Any],
) -> None:

    print()
    print("-" * 78)
    print(name)
    print("-" * 78)

    print(
        f"Threshold : {metrics['threshold']:.3f}"
    )

    print(
        f"ROC-AUC   : {metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : {metrics['pr_auc']:.4f}"
    )

    print(
        f"F1        : {metrics['f1']:.4f}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"TN={metrics['tn']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TP={metrics['tp']}"
    )


# =============================================================================
# RUN V1
# =============================================================================

def evaluate_v1() -> Dict[str, Any]:

    print()
    print("=" * 78)
    print("V1 EVALUATION")
    print("=" * 78)

    checkpoint_path = CHECKPOINTS[
        "v1"
    ]

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    test_dataset, _time_stats = (
        load_v1_test_data()
    )

    model = create_model(
        "v1"
    )

    checkpoint = load_checkpoint(
        model,
        checkpoint_path,
        DEVICE,
    )

    threshold = get_checkpoint_threshold(
        checkpoint
    )

    epoch = get_checkpoint_epoch(
        checkpoint
    )

    val_f1 = get_checkpoint_val_f1(
        checkpoint
    )

    y_true, y_prob = (
        run_v1_inference(
            model,
            test_dataset,
        )
    )

    metrics = classification_metrics(
        y_true,
        y_prob,
        threshold,
    )

    best_f1 = find_best_f1_threshold(
        y_true,
        y_prob,
    )

    print(
        f"Test samples : {len(y_true)}"
    )

    if epoch is not None:
        print(
            f"Checkpoint epoch : {epoch}"
        )

    if val_f1 is not None:
        print(
            f"Validation F1   : {val_f1:.4f}"
        )

    print_metrics(
        "V1 TEST",
        metrics,
    )

    print()
    print(
        "Fine-grained test threshold diagnostic:"
    )

    print(
        f"  Best test F1 threshold : "
        f"{best_f1['threshold']:.3f}"
    )

    print(
        f"  Best test F1           : "
        f"{best_f1['f1']:.4f}"
    )

    print(
        f"  Precision              : "
        f"{best_f1['precision']:.4f}"
    )

    print(
        f"  Recall                 : "
        f"{best_f1['recall']:.4f}"
    )

    return {
        "model": "v1",
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_epoch": epoch,
        "checkpoint_val_f1": val_f1,
        "test_metrics": metrics,
        "best_test_f1_diagnostic": best_f1,
    }


# =============================================================================
# RUN STANDARD V2 / V3-A / V3-C
# =============================================================================

def evaluate_standard_model(
    model_name: str,
) -> Dict[str, Any]:

    print()
    print("=" * 78)
    print(
        f"{model_name.upper()} EVALUATION"
    )
    print("=" * 78)

    checkpoint_path = CHECKPOINTS[
        model_name
    ]

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    test_dataset, time_stats = (
        load_v2_test_data()
    )

    model = create_model(
        model_name
    )

    checkpoint = load_checkpoint(
        model,
        checkpoint_path,
        DEVICE,
    )

    official_threshold = (
        get_checkpoint_threshold(
            checkpoint
        )
    )

    epoch = get_checkpoint_epoch(
        checkpoint
    )

    val_f1 = get_checkpoint_val_f1(
        checkpoint
    )

    (
        y_true,
        original_probabilities,
        category_ids,
        temporal_features,
        raw_delta_hours,
    ) = run_standard_inference(
        model,
        test_dataset,
    )

    original_metrics = classification_metrics(
        y_true,
        original_probabilities,
        official_threshold,
    )

    best_test_f1 = find_best_f1_threshold(
        y_true,
        original_probabilities,
    )

    print(
        f"Test samples : {len(y_true)}"
    )

    if epoch is not None:
        print(
            f"Checkpoint epoch : {epoch}"
        )

    if val_f1 is not None:
        print(
            f"Validation F1   : {val_f1:.4f}"
        )

    print_metrics(
        f"{model_name.upper()} ORIGINAL TEST",
        original_metrics,
    )

    print()
    print(
        "Fine-grained test threshold diagnostic:"
    )

    print(
        f"  Best test F1 threshold : "
        f"{best_test_f1['threshold']:.3f}"
    )

    print(
        f"  Best test F1           : "
        f"{best_test_f1['f1']:.4f}"
    )

    # -------------------------------------------------------------------------
    # OPTION A
    # -------------------------------------------------------------------------

    print()
    print(
        "Running Option A: remove recurrence information..."
    )

    option_a_features = (
        make_option_a_temporal_features(
            temporal_features,
            time_stats,
        )
    )

    option_a_probabilities = (
        run_standard_with_modified_features(
            model,
            category_ids,
            option_a_features,
            raw_delta_hours,
        )
    )

    option_a_metrics = classification_metrics(
        y_true,
        option_a_probabilities,
        official_threshold,
    )

    option_a_best_f1 = find_best_f1_threshold(
        y_true,
        option_a_probabilities,
    )

    option_a_delta = (
        probability_delta_summary(
            original_probabilities,
            option_a_probabilities,
        )
    )

    print_metrics(
        f"{model_name.upper()} OPTION A",
        option_a_metrics,
    )

    print()
    print(
        "Option A probability delta:"
    )

    for key, value in option_a_delta.items():
        print(
            f"  {key:24s}: {value:.6f}"
        )

    print()
    print(
        "Option A best-test-F1 diagnostic:"
    )

    print(
        f"  threshold : "
        f"{option_a_best_f1['threshold']:.3f}"
    )

    print(
        f"  F1        : "
        f"{option_a_best_f1['f1']:.4f}"
    )

    print(
        f"  Precision : "
        f"{option_a_best_f1['precision']:.4f}"
    )

    print(
        f"  Recall    : "
        f"{option_a_best_f1['recall']:.4f}"
    )

    # -------------------------------------------------------------------------
    # OPTION B
    # -------------------------------------------------------------------------

    print()
    print(
        "Running Option B: globally shuffle recurrence pairs..."
    )

    option_b_features = (
        make_option_b_temporal_features(
            temporal_features,
            seed=OPTION_B_SEED,
        )
    )

    option_b_probabilities = (
        run_standard_with_modified_features(
            model,
            category_ids,
            option_b_features,
            raw_delta_hours,
        )
    )

    option_b_metrics = classification_metrics(
        y_true,
        option_b_probabilities,
        official_threshold,
    )

    option_b_best_f1 = find_best_f1_threshold(
        y_true,
        option_b_probabilities,
    )

    option_b_delta = (
        probability_delta_summary(
            original_probabilities,
            option_b_probabilities,
        )
    )

    print_metrics(
        f"{model_name.upper()} OPTION B",
        option_b_metrics,
    )

    print()
    print(
        "Option B probability delta:"
    )

    for key, value in option_b_delta.items():
        print(
            f"  {key:24s}: {value:.6f}"
        )

    print()
    print(
        "Option B best-test-F1 diagnostic:"
    )

    print(
        f"  threshold : "
        f"{option_b_best_f1['threshold']:.3f}"
    )

    print(
        f"  F1        : "
        f"{option_b_best_f1['f1']:.4f}"
    )

    print(
        f"  Precision : "
        f"{option_b_best_f1['precision']:.4f}"
    )

    print(
        f"  Recall    : "
        f"{option_b_best_f1['recall']:.4f}"
    )

    return {
        "model": model_name,
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_epoch": epoch,
        "checkpoint_val_f1": val_f1,
        "official_threshold": official_threshold,
        "test_samples": int(
            len(y_true)
        ),
        "original": {
            "metrics": original_metrics,
            "best_test_f1_diagnostic": best_test_f1,
        },
        "option_a": {
            "metrics": option_a_metrics,
            "best_test_f1_diagnostic": option_a_best_f1,
            "probability_delta": option_a_delta,
        },
        "option_b": {
            "metrics": option_b_metrics,
            "best_test_f1_diagnostic": option_b_best_f1,
            "probability_delta": option_b_delta,
        },
    }


# =============================================================================
# RUN BUCKETED MODEL
# =============================================================================

def evaluate_bucketed() -> Dict[str, Any]:

    model_name = "v3a_bucketed"

    print()
    print("=" * 78)
    print("V3-A BUCKETED RECURRENCE EVALUATION")
    print("=" * 78)

    checkpoint_path = CHECKPOINTS[
        model_name
    ]

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    test_dataset, _time_stats = (
        load_bucketed_test_data()
    )

    model = create_model(
        model_name
    )

    checkpoint = load_checkpoint(
        model,
        checkpoint_path,
        DEVICE,
    )

    official_threshold = (
        get_checkpoint_threshold(
            checkpoint
        )
    )

    epoch = get_checkpoint_epoch(
        checkpoint
    )

    val_f1 = get_checkpoint_val_f1(
        checkpoint
    )

    (
        y_true,
        original_probabilities,
        category_ids,
        temporal_features,
        raw_delta_hours,
        bucket_ids,
    ) = run_bucketed_inference(
        model,
        test_dataset,
    )

    original_metrics = classification_metrics(
        y_true,
        original_probabilities,
        official_threshold,
    )

    best_test_f1 = find_best_f1_threshold(
        y_true,
        original_probabilities,
    )

    print(
        f"Test samples : {len(y_true)}"
    )

    if epoch is not None:
        print(
            f"Checkpoint epoch : {epoch}"
        )

    if val_f1 is not None:
        print(
            f"Validation F1   : {val_f1:.4f}"
        )

    print_metrics(
        "V3-A BUCKETED ORIGINAL TEST",
        original_metrics,
    )

    print()
    print(
        "Fine-grained test threshold diagnostic:"
    )

    print(
        f"  Best test F1 threshold : "
        f"{best_test_f1['threshold']:.3f}"
    )

    print(
        f"  Best test F1           : "
        f"{best_test_f1['f1']:.4f}"
    )

    # -------------------------------------------------------------------------
    # BUCKETED OPTION B
    # -------------------------------------------------------------------------

    print()
    print(
        "Running Bucketed Option B: "
        "globally shuffle recurrence buckets..."
    )

    shuffled_bucket_ids = (
        make_bucketed_option_b(
            bucket_ids,
            seed=OPTION_B_SEED,
        )
    )

    option_b_probabilities = (
        run_bucketed_with_modified_features(
            model,
            category_ids,
            temporal_features,
            raw_delta_hours,
            shuffled_bucket_ids,
        )
    )

    option_b_metrics = classification_metrics(
        y_true,
        option_b_probabilities,
        official_threshold,
    )

    option_b_best_f1 = find_best_f1_threshold(
        y_true,
        option_b_probabilities,
    )

    option_b_delta = (
        probability_delta_summary(
            original_probabilities,
            option_b_probabilities,
        )
    )

    print_metrics(
        "V3-A BUCKETED OPTION B",
        option_b_metrics,
    )

    print()
    print(
        "Bucketed Option B probability delta:"
    )

    for key, value in option_b_delta.items():
        print(
            f"  {key:24s}: {value:.6f}"
        )

    print()
    print(
        "Bucketed Option B best-test-F1 diagnostic:"
    )

    print(
        f"  threshold : "
        f"{option_b_best_f1['threshold']:.3f}"
    )

    print(
        f"  F1        : "
        f"{option_b_best_f1['f1']:.4f}"
    )

    print(
        f"  Precision : "
        f"{option_b_best_f1['precision']:.4f}"
    )

    print(
        f"  Recall    : "
        f"{option_b_best_f1['recall']:.4f}"
    )

    return {
        "model": model_name,
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_epoch": epoch,
        "checkpoint_val_f1": val_f1,
        "official_threshold": official_threshold,
        "test_samples": int(
            len(y_true)
        ),
        "original": {
            "metrics": original_metrics,
            "best_test_f1_diagnostic": best_test_f1,
        },
        "option_b": {
            "metrics": option_b_metrics,
            "best_test_f1_diagnostic": option_b_best_f1,
            "probability_delta": option_b_delta,
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    set_seed(
        GLOBAL_SEED
    )

    print()
    print("=" * 78)
    print(
        "SOCIA -- RECURRENCE ABLATION EVALUATION"
    )
    print("=" * 78)

    print()
    print(
        f"Device : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Global seed      : {GLOBAL_SEED}"
    )

    print(
        f"Option B seed    : {OPTION_B_SEED}"
    )

    print()
    print(
        "Models:"
    )

    for name, path in CHECKPOINTS.items():

        print(
            f"  {name:18s} -> {path}"
        )

    # -------------------------------------------------------------------------
    # Results container
    # -------------------------------------------------------------------------

    results: Dict[str, Any] = {
        "device": str(DEVICE),
        "global_seed": GLOBAL_SEED,
        "option_b_seed": OPTION_B_SEED,
        "models": {},
    }

    # -------------------------------------------------------------------------
    # V1
    #
    # V1 is evaluated on its original benchmark.
    # It is NOT subjected to the V2 recurrence-channel ablations.
    # -------------------------------------------------------------------------

    results["models"]["v1"] = (
        evaluate_v1()
    )

    # -------------------------------------------------------------------------
    # V2
    # -------------------------------------------------------------------------

    results["models"]["v2"] = (
        evaluate_standard_model(
            "v2"
        )
    )

    # -------------------------------------------------------------------------
    # V3-A
    # -------------------------------------------------------------------------

    results["models"]["v3a"] = (
        evaluate_standard_model(
            "v3a"
        )
    )

    # -------------------------------------------------------------------------
    # V3-C
    # -------------------------------------------------------------------------

    results["models"]["v3c"] = (
        evaluate_standard_model(
            "v3c"
        )
    )

    # -------------------------------------------------------------------------
    # V3-A Bucketed
    # -------------------------------------------------------------------------

    results["models"]["v3a_bucketed"] = (
        evaluate_bucketed()
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    output_path = (
        ROOT
        / "recurrence_ablation_all_results.json"
    )

    save_json(
        output_path,
        results,
    )

    print()
    print("=" * 78)
    print(
        "EVALUATION COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        f"Results saved to:"
    )

    print(
        output_path
    )

    print()


if __name__ == "__main__":
    main()