"""
Socia V2 Training

Same overall protocol as train.py (seed, optimizer, warmup+cosine
schedule, threshold-sweep checkpoint selection, early stopping). What's
new is the training loss, which adds two auxiliary terms on top of
standard BCE, applied only to the POSITIVE examples in each batch:

1. Pairwise ranking loss (primary lever):
   For each positive window, build a "too-slow" counterpart by
   stretching its inter-event gaps by a factor sampled per-example
   from [TOO_SLOW_MIN_FACTOR, TOO_SLOW_MAX_FACTOR] (covers realistic
   near-miss stretches like 1.5x-2x as well as more extreme ones).
   Category identities and event order are exactly preserved -- only
   timing changes. The model is pushed to score the original higher
   than its stretched counterpart by at least RANKING_MARGIN, via a
   hinge loss.

2. Reordered-negative auxiliary loss (secondary, lower weight):
   For each positive window, build a counterpart with the SAME
   timestamps but a randomly permuted assignment of categories to
   those timestamps, trained as label 0. Kept at low weight -- V1's
   order-shuffle ablation showed the OLD architecture couldn't use
   this signal at all; V2's relative-time attention bias gives it a
   mechanism to, but the ranking loss on too-slow pairs is the
   primary, better-motivated lever.

Validation is standard/unaugmented -- no ranking or reorder loss is
applied during validation, and checkpoint selection is still by plain
val F1 via the same threshold sweep as V1.

---------------------------------------------------------------------
NaN FIX: run_validation now calls the model with check_finite=True.
This was the pathway that produced NaN val_loss in the previous
version, caused by nn.TransformerEncoder's fused fast-path kernel
being incompatible with a custom per-head float attention mask under
torch.no_grad(). See model_v2.py's module docstring for the full
diagnosis. The underlying fix is architectural (model_v2.py no longer
uses nn.TransformerEncoder at all); check_finite=True here is a
safety net, not the fix itself -- if it ever fires again, it will
now tell you exactly which stage broke, rather than surfacing only
as a NaN loss three functions away from the cause.
---------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

from dataset_v2 import (
    TEMPORAL_FEATURE_DIM,
    build_datasets_v2,
    compute_temporal_features_single,
)
from dataset import create_dataloader  # unchanged, generic over any Dataset
from model_v2 import SociaPatternTransformerV2


# =====================================================================
# CONFIG -- unchanged from V1 unless noted
# =====================================================================

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 8
WARMUP_FRACTION = 0.10

THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.05

# --- V2 training-augmentation hyperparameters ---

TOO_SLOW_MIN_FACTOR = 1.3
TOO_SLOW_MAX_FACTOR = 5.0

RANKING_MARGIN = 0.5
RANKING_LOSS_WEIGHT = 1.0

REORDER_LOSS_WEIGHT = 0.3

OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints"
BEST_CHECKPOINT_PATH = OUTPUT_DIR / "best_model_v2.pt"
HISTORY_PATH = OUTPUT_DIR / "training_history_v2.json"


# =====================================================================
# PRESERVED FROM V1 (unchanged logic)
# =====================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"Device: {device}")
    return device


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float = WARMUP_FRACTION,
):
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_thresholds() -> List[float]:
    thresholds = []
    current = THRESHOLD_MIN
    while current <= THRESHOLD_MAX + 1e-9:
        thresholds.append(round(current, 2))
        current += THRESHOLD_STEP
    return thresholds


# =====================================================================
# TRAINING-TIME AUGMENTATION HELPERS
# =====================================================================

def stretch_raw_delta_hours(raw_hours: List[float], factor: float) -> List[float]:
    """
    Stretch inter-event gaps by `factor`, preserving the first
    timestamp. Assumes raw_hours is already non-decreasing (guaranteed
    by dataset validation), so no defensive sorting is needed here.
    """

    if len(raw_hours) <= 1:
        return list(raw_hours)

    result = [raw_hours[0]]
    for i in range(1, len(raw_hours)):
        gap = raw_hours[i] - raw_hours[i - 1]
        result.append(result[-1] + gap * factor)

    return result


def build_too_slow_batch(
    category_ids_subset: torch.Tensor,  # (K, L) long, CPU
    raw_delta_hours_subset: torch.Tensor,  # (K, L) float, CPU
    time_stats: Dict[str, float],
    min_factor: float,
    max_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build temporally-stretched counterparts for a batch of positive
    windows. Category identities and order are preserved exactly;
    only inter-event gaps are stretched, by a factor sampled
    independently per example.
    """

    K = category_ids_subset.size(0)

    category_out = category_ids_subset.clone()
    temporal_list = []
    raw_out_list = []

    for k in range(K):
        cats = category_ids_subset[k].tolist()
        raw_hours = raw_delta_hours_subset[k].tolist()

        factor = random.uniform(min_factor, max_factor)
        stretched_hours = stretch_raw_delta_hours(raw_hours, factor)

        features = compute_temporal_features_single(cats, stretched_hours, time_stats)

        temporal_list.append(features)
        raw_out_list.append(torch.tensor(stretched_hours, dtype=torch.float32))

    temporal_out = torch.stack(temporal_list, dim=0)
    raw_out = torch.stack(raw_out_list, dim=0)

    return category_out, temporal_out, raw_out


def build_reordered_batch(
    category_ids_subset: torch.Tensor,  # (K, L) long, CPU
    raw_delta_hours_subset: torch.Tensor,  # (K, L) float, CPU
    time_stats: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build order-shuffled counterparts: timestamps unchanged, category
    assignment to those timestamps randomly permuted. Treated as
    label-0 training examples.
    """

    K = category_ids_subset.size(0)

    category_list = []
    temporal_list = []
    raw_out_list = []

    for k in range(K):
        cats = category_ids_subset[k].tolist()
        raw_hours = raw_delta_hours_subset[k].tolist()

        permuted_indices = list(range(len(cats)))
        random.shuffle(permuted_indices)
        shuffled_cats = [cats[i] for i in permuted_indices]

        features = compute_temporal_features_single(shuffled_cats, raw_hours, time_stats)

        category_list.append(torch.tensor(shuffled_cats, dtype=torch.long))
        temporal_list.append(features)
        raw_out_list.append(torch.tensor(raw_hours, dtype=torch.float32))

    category_out = torch.stack(category_list, dim=0)
    temporal_out = torch.stack(temporal_list, dim=0)
    raw_out = torch.stack(raw_out_list, dim=0)

    return category_out, temporal_out, raw_out


# =====================================================================
# VALIDATION -- unaugmented, same protocol as V1, with finite checks
# =====================================================================

def run_validation(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:

    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for category_ids, temporal_features, raw_delta_hours, labels in loader:

            category_ids = category_ids.to(device)
            temporal_features = temporal_features.to(device)
            raw_delta_hours = raw_delta_hours.to(device)
            labels = labels.to(device)

            # check_finite=True: this is the exact code path that
            # previously produced NaN under torch.no_grad(). Left on
            # here as a safety net -- if it ever fires again, it
            # raises immediately with the specific stage name instead
            # of surfacing as an unexplained NaN in val_loss.
            logits = model(
                category_ids,
                temporal_features,
                raw_delta_hours,
                check_finite=True,
            )

            loss = loss_fn(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            probs = torch.sigmoid(logits)

            all_labels.extend(labels.long().cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / total_examples

    thresholds = get_thresholds()

    best_threshold = 0.50
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = -1.0

    for threshold in thresholds:
        preds = [1 if prob >= threshold else 0 for prob in all_probs]

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average="binary", zero_division=0
        )

        precision, recall, f1 = float(precision), float(recall), float(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return {
        "loss": float(avg_loss),
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
        "threshold": best_threshold,
    }


# =====================================================================
# TRAINING
# =====================================================================

def train() -> None:

    set_seed(SEED)
    device = get_device()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, time_stats = build_datasets_v2()

    train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Time stats (V2): {time_stats}")
    print(
        f"Too-slow ranking loss: weight={RANKING_LOSS_WEIGHT}, "
        f"margin={RANKING_MARGIN}, "
        f"stretch=[{TOO_SLOW_MIN_FACTOR}, {TOO_SLOW_MAX_FACTOR}]"
    )
    print(f"Reorder auxiliary loss: weight={REORDER_LOSS_WEIGHT}")

    model = SociaPatternTransformerV2().to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = make_lr_scheduler(optimizer, total_steps=total_steps, warmup_fraction=WARMUP_FRACTION)

    history: List[Dict[str, float]] = []

    best_val_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.50
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        running_total_loss = 0.0
        running_main_loss = 0.0
        running_ranking_loss = 0.0
        running_reorder_loss = 0.0
        running_examples = 0

        for category_ids, temporal_features, raw_delta_hours, labels in train_loader:

            category_ids = category_ids.to(device)
            temporal_features = temporal_features.to(device)
            raw_delta_hours = raw_delta_hours.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(category_ids, temporal_features, raw_delta_hours)
            main_loss = loss_fn(logits, labels)

            positive_mask = labels == 1
            num_positive = int(positive_mask.sum().item())

            ranking_loss = torch.tensor(0.0, device=device)
            reorder_loss = torch.tensor(0.0, device=device)

            if num_positive > 0:

                pos_category_cpu = category_ids[positive_mask].detach().cpu()
                pos_raw_cpu = raw_delta_hours[positive_mask].detach().cpu()

                # --- Too-slow pairwise ranking loss ---
                slow_category, slow_temporal, slow_raw = build_too_slow_batch(
                    pos_category_cpu, pos_raw_cpu, time_stats,
                    TOO_SLOW_MIN_FACTOR, TOO_SLOW_MAX_FACTOR,
                )
                slow_category = slow_category.to(device)
                slow_temporal = slow_temporal.to(device)
                slow_raw = slow_raw.to(device)

                logits_slow = model(slow_category, slow_temporal, slow_raw)
                logits_orig_pos = logits[positive_mask]

                hinge = torch.relu(RANKING_MARGIN - (logits_orig_pos - logits_slow))
                ranking_loss = hinge.mean()

                # --- Reordered negative auxiliary loss ---
                if REORDER_LOSS_WEIGHT > 0:
                    reorder_category, reorder_temporal, reorder_raw = build_reordered_batch(
                        pos_category_cpu, pos_raw_cpu, time_stats,
                    )
                    reorder_category = reorder_category.to(device)
                    reorder_temporal = reorder_temporal.to(device)
                    reorder_raw = reorder_raw.to(device)

                    logits_reorder = model(reorder_category, reorder_temporal, reorder_raw)
                    reorder_targets = torch.zeros_like(logits_reorder)
                    reorder_loss = loss_fn(logits_reorder, reorder_targets)

            total_loss = (
                main_loss
                + RANKING_LOSS_WEIGHT * ranking_loss
                + REORDER_LOSS_WEIGHT * reorder_loss
            )

            total_loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            batch_size = labels.size(0)
            running_total_loss += total_loss.item() * batch_size
            running_main_loss += main_loss.item() * batch_size
            running_ranking_loss += float(ranking_loss.item()) * batch_size
            running_reorder_loss += float(reorder_loss.item()) * batch_size
            running_examples += batch_size

        train_total_loss = running_total_loss / running_examples
        train_main_loss = running_main_loss / running_examples
        train_ranking_loss = running_ranking_loss / running_examples
        train_reorder_loss = running_reorder_loss / running_examples

        val_metrics = run_validation(model, val_loader, loss_fn, device)

        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | lr={current_lr:.6f} | "
            f"train_total={train_total_loss:.4f} "
            f"(main={train_main_loss:.4f}, rank={train_ranking_loss:.4f}, "
            f"reorder={train_reorder_loss:.4f}) | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"best_threshold={val_metrics['threshold']:.2f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_total_loss": train_total_loss,
            "train_main_loss": train_main_loss,
            "train_ranking_loss": train_ranking_loss,
            "train_reorder_loss": train_reorder_loss,
            "val_loss": val_metrics["loss"],
            "val_threshold": val_metrics["threshold"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        })

        if val_metrics["f1"] > best_val_f1:

            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_threshold = val_metrics["threshold"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_val_f1,
                    "val_threshold": best_threshold,
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "time_stats": time_stats,
                    "seed": SEED,
                    "model_version": "v2",
                    "augmentation_config": {
                        "too_slow_min_factor": TOO_SLOW_MIN_FACTOR,
                        "too_slow_max_factor": TOO_SLOW_MAX_FACTOR,
                        "ranking_margin": RANKING_MARGIN,
                        "ranking_loss_weight": RANKING_LOSS_WEIGHT,
                        "reorder_loss_weight": REORDER_LOSS_WEIGHT,
                    },
                },
                BEST_CHECKPOINT_PATH,
            )

            print(f"  -> New best val_f1={best_val_f1:.4f} at threshold={best_threshold:.2f}, checkpoint saved.")

        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement}/{EARLY_STOP_PATIENCE} epochs.")

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best epoch was {best_epoch}, best val_f1={best_val_f1:.4f}, "
                    f"threshold={best_threshold:.2f})."
                )
                break

    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Best val_f1: {best_val_f1:.4f}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best checkpoint saved to: {BEST_CHECKPOINT_PATH}")
    print(f"Training history saved to: {HISTORY_PATH}")


if __name__ == "__main__":
    train()