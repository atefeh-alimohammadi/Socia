from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

from dataset import build_datasets, create_dataloader
from model import SociaPatternTransformer


SEED = 42
BATCH_SIZE = 64

# Keep the lower LR for this experiment so that the only major
# methodological change is threshold-optimized validation selection.
LEARNING_RATE = 5e-5

WEIGHT_DECAY = 0.01
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 8
WARMUP_FRACTION = 0.10

# Validation thresholds used for checkpoint selection.
THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.05


OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints"
BEST_CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"


def set_seed(seed: int = SEED) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            f"Device: {device} "
            f"({torch.cuda.get_device_name(0)})"
        )

    else:

        device = torch.device("cpu")

        print(f"Device: {device}")

    return device


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float = WARMUP_FRACTION,
):
    """
    Linear warmup for the first `warmup_fraction` of total steps,
    followed by cosine decay to 0.
    """

    warmup_steps = max(
        1,
        int(total_steps * warmup_fraction),
    )

    def lr_lambda(current_step: int) -> float:

        if current_step < warmup_steps:

            return current_step / max(
                1,
                warmup_steps,
            )

        progress = (
            current_step - warmup_steps
        ) / max(
            1,
            total_steps - warmup_steps,
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


def get_thresholds() -> List[float]:

    thresholds = []

    current = THRESHOLD_MIN

    while current <= THRESHOLD_MAX + 1e-9:

        thresholds.append(
            round(current, 2)
        )

        current += THRESHOLD_STEP

    return thresholds


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

        for category_ids, delta_hours, labels in loader:

            category_ids = category_ids.to(device)

            delta_hours = delta_hours.to(device)

            labels = labels.to(device)

            logits = model(
                category_ids,
                delta_hours,
            )

            loss = loss_fn(
                logits,
                labels,
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_examples += batch_size

            probs = torch.sigmoid(logits)

            all_labels.extend(
                labels.long()
                .cpu()
                .tolist()
            )

            all_probs.extend(
                probs.cpu().tolist()
            )

    avg_loss = total_loss / total_examples

    thresholds = get_thresholds()

    best_threshold = 0.50
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = -1.0

    for threshold in thresholds:

        preds = [
            1 if prob >= threshold else 0
            for prob in all_probs
        ]

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                all_labels,
                preds,
                average="binary",
                zero_division=0,
            )
        )

        precision = float(precision)

        recall = float(recall)

        f1 = float(f1)

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


def train() -> None:

    set_seed(SEED)

    device = get_device()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
        time_stats,
    ) = build_datasets()

    train_loader = create_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Val batches:   {len(val_loader)}"
    )

    print(
        f"Time stats:    {time_stats}"
    )

    model = SociaPatternTransformer().to(
        device
    )

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = (
        len(train_loader)
        * MAX_EPOCHS
    )

    scheduler = make_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=WARMUP_FRACTION,
    )

    history: List[Dict[str, float]] = []

    best_val_f1 = -1.0

    best_epoch = -1

    best_threshold = 0.50

    epochs_without_improvement = 0

    global_step = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        running_examples = 0

        for (
            category_ids,
            delta_hours,
            labels,
        ) in train_loader:

            category_ids = category_ids.to(
                device
            )

            delta_hours = delta_hours.to(
                device
            )

            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(
                category_ids,
                delta_hours,
            )

            loss = loss_fn(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            scheduler.step()

            global_step += 1

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            running_examples += batch_size

        train_loss = (
            running_loss
            / running_examples
        )

        val_metrics = run_validation(
            model,
            val_loader,
            loss_fn,
            device,
        )

        current_lr = (
            scheduler.get_last_lr()[0]
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"best_threshold={val_metrics['threshold']:.2f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_threshold": val_metrics[
                    "threshold"
                ],
                "val_precision": val_metrics[
                    "precision"
                ],
                "val_recall": val_metrics[
                    "recall"
                ],
                "val_f1": val_metrics["f1"],
            }
        )

        if (
            val_metrics["f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["f1"]
            )

            best_epoch = epoch

            best_threshold = (
                val_metrics["threshold"]
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "val_f1": best_val_f1,
                    "val_threshold": (
                        best_threshold
                    ),
                    "val_precision": (
                        val_metrics["precision"]
                    ),
                    "val_recall": (
                        val_metrics["recall"]
                    ),
                    "time_stats": time_stats,
                    "seed": SEED,
                },
                BEST_CHECKPOINT_PATH,
            )

            print(
                f"  -> New best "
                f"val_f1={best_val_f1:.4f} "
                f"at threshold="
                f"{best_threshold:.2f}, "
                f"checkpoint saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  -> No improvement for "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOP_PATIENCE} epochs."
            )

            if (
                epochs_without_improvement
                >= EARLY_STOP_PATIENCE
            ):

                print(
                    f"Early stopping triggered "
                    f"at epoch {epoch} "
                    f"(best epoch was "
                    f"{best_epoch}, "
                    f"best val_f1="
                    f"{best_val_f1:.4f}, "
                    f"threshold="
                    f"{best_threshold:.2f})."
                )

                break

    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    print()

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best val_f1: {best_val_f1:.4f}"
    )

    print(
        f"Best threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best checkpoint saved to: "
        f"{BEST_CHECKPOINT_PATH}"
    )

    print(
        f"Training history saved to: "
        f"{HISTORY_PATH}"
    )

    print()

    print(
        "Restoring best checkpoint "
        "into model..."
    )

    checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Restored model from epoch "
        f"{checkpoint['epoch']} "
        f"with val_f1="
        f"{checkpoint['val_f1']:.4f} "
        f"at threshold="
        f"{checkpoint.get('val_threshold', 0.50):.2f}."
    )


if __name__ == "__main__":

    train()