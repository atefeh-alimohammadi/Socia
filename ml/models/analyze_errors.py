from __future__ import annotations

import json
from pathlib import Path

import torch

from dataset import build_datasets, create_dataloader
from model import SociaPatternTransformer


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "best_model.pt"
)

THRESHOLD = 0.30
TOP_N = 30


def load_jsonl(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    print("\nLoading validation data...")

    val_records = load_jsonl(DATA_DIR / "val.jsonl")
    metadata_records = load_jsonl(
        DATA_DIR / "val_eval_metadata.jsonl"
    )

    print(f"Validation records: {len(val_records)}")
    print(f"Metadata records:   {len(metadata_records)}")

    assert len(val_records) == len(metadata_records), (
        "val.jsonl and val_eval_metadata.jsonl have different lengths"
    )

    (
        _train_dataset,
        val_dataset,
        _test_dataset,
        _time_stats,
    ) = build_datasets()

    loader = create_dataloader(
        val_dataset,
        batch_size=64,
        shuffle=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model = SociaPatternTransformer().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    results = []

    record_index = 0

    with torch.no_grad():

        for category_ids, delta_hours, labels in loader:

            category_ids = category_ids.to(device)
            delta_hours = delta_hours.to(device)

            logits = model(
                category_ids,
                delta_hours,
            )

            probabilities = torch.sigmoid(logits)

            for i in range(len(labels)):

                probability = float(
                    probabilities[i].cpu().item()
                )

                true_label = int(
                    labels[i].cpu().item()
                )

                predicted_label = (
                    1
                    if probability >= THRESHOLD
                    else 0
                )

                record = val_records[record_index]
                metadata = metadata_records[record_index]

                results.append(
                    {
                        "index": record_index,
                        "window_id": record.get("window_id"),
                        "true_label": true_label,
                        "probability": probability,
                        "predicted_label": predicted_label,
                        "error_type": (
                            "TP"
                            if true_label == 1
                            and predicted_label == 1
                            else "TN"
                            if true_label == 0
                            and predicted_label == 0
                            else "FP"
                            if true_label == 0
                            and predicted_label == 1
                            else "FN"
                        ),
                        "events": record.get("events"),
                        "true_pattern_id": metadata.get(
                            "true_pattern_id"
                        ),
                        "is_interleaved_occurrence": metadata.get(
                            "is_interleaved_occurrence"
                        ),
                        "window_relative_true_positions": metadata.get(
                            "window_relative_true_positions"
                        ),
                    }
                )

                record_index += 1

    assert record_index == len(val_records), (
        f"Processed {record_index} records but expected "
        f"{len(val_records)}"
    )

    false_positives = [
        r for r in results
        if r["error_type"] == "FP"
    ]

    false_negatives = [
        r for r in results
        if r["error_type"] == "FN"
    ]

    true_positives = [
        r for r in results
        if r["error_type"] == "TP"
    ]

    true_negatives = [
        r for r in results
        if r["error_type"] == "TN"
    ]

    print()
    print("=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)

    print(f"\nThreshold: {THRESHOLD}")

    print("\nCounts:")
    print(f"  TP: {len(true_positives)}")
    print(f"  TN: {len(true_negatives)}")
    print(f"  FP: {len(false_positives)}")
    print(f"  FN: {len(false_negatives)}")

    print()
    print("=" * 80)
    print("TOP FALSE POSITIVES")
    print("=" * 80)

    # Highest-confidence false positives first.
    false_positives.sort(
        key=lambda x: x["probability"],
        reverse=True,
    )

    for rank, item in enumerate(
        false_positives[:TOP_N],
        start=1,
    ):

        print()
        print(f"--- FP #{rank} ---")
        print(f"index: {item['index']}")
        print(f"window_id: {item['window_id']}")
        print(f"probability: {item['probability']:.4f}")
        print(f"true_label: {item['true_label']}")
        print(f"true_pattern_id: {item['true_pattern_id']}")
        print(
            "is_interleaved_occurrence: "
            f"{item['is_interleaved_occurrence']}"
        )
        print(
            "true_positions: "
            f"{item['window_relative_true_positions']}"
        )

        categories = [
            event.get("category")
            for event in item["events"]
        ]

        deltas = [
            event.get("delta_hours")
            for event in item["events"]
        ]

        print(f"categories: {categories}")
        print(f"delta_hours: {deltas}")

    print()
    print("=" * 80)
    print("TOP FALSE NEGATIVES")
    print("=" * 80)

    # Lowest-confidence false negatives first.
    false_negatives.sort(
        key=lambda x: x["probability"]
    )

    for rank, item in enumerate(
        false_negatives[:TOP_N],
        start=1,
    ):

        print()
        print(f"--- FN #{rank} ---")
        print(f"index: {item['index']}")
        print(f"window_id: {item['window_id']}")
        print(f"probability: {item['probability']:.4f}")
        print(f"true_label: {item['true_label']}")
        print(f"true_pattern_id: {item['true_pattern_id']}")
        print(
            "is_interleaved_occurrence: "
            f"{item['is_interleaved_occurrence']}"
        )
        print(
            "true_positions: "
            f"{item['window_relative_true_positions']}"
        )

        categories = [
            event.get("category")
            for event in item["events"]
        ]

        deltas = [
            event.get("delta_hours")
            for event in item["events"]
        ]

        print(f"categories: {categories}")
        print(f"delta_hours: {deltas}")

    # Save everything for easier inspection.
    output_path = (
        Path(__file__).resolve().parent
        / "error_analysis_val.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "threshold": THRESHOLD,
                "counts": {
                    "TP": len(true_positives),
                    "TN": len(true_negatives),
                    "FP": len(false_positives),
                    "FN": len(false_negatives),
                },
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_positives": true_positives[:50],
                "true_negatives": true_negatives[:50],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"\nFull analysis saved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()