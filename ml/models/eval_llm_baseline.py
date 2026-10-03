import json
import re
import time
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

TEST_PATH = Path(
    r"D:\AI Companion\Socia\ml\dataset\scripts\data\test.jsonl"
)

MODEL = "qwen2.5:7b"

OLLAMA_URL = "http://localhost:11434/api/generate"

OUTPUT_PATH = Path(
    r"D:\AI Companion\Socia\ml\eval\eval_outputs\llm_baseline_results.json"
)


MAX_SAMPLES = None


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a behavioral pattern detection system.

Your task is to determine whether the given event sequence
contains a recurring behavioral pattern.

A positive example means:

- the same ordered behavioral category pattern occurs
- at least twice
- in temporally separated occurrences/sites
- and the occurrences are well-formed.

A negative example means that this recurrence criterion is not satisfied.

Return ONLY valid JSON:

{
  "label": 0 or 1,
  "confidence": number between 0 and 1
}

Do not output any explanation.
"""


def build_prompt(events):
    return f"""
{SYSTEM_PROMPT}

Event sequence:

{json.dumps(events, ensure_ascii=False, indent=2)}

Does this sequence contain the defined recurring behavioral pattern?
"""


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(events):
    prompt = build_prompt(events)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    raw = data["response"]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # fallback if model returns slightly malformed JSON
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if not match:
            raise ValueError(
                f"Could not parse LLM response:\n{raw}"
            )

        result = json.loads(match.group(0))

    label = int(result["label"])
    confidence = float(result["confidence"])

    if label not in (0, 1):
        raise ValueError(f"Invalid label: {label}")

    confidence = max(0.0, min(1.0, confidence))

    return label, confidence, raw


# ============================================================
# DATASET LOADING
# ============================================================

def load_test_set(path):
    samples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            samples.append(item)

    return samples


def extract_events(item):
    """
    Adapt this function if your V2 JSONL uses a different field name.
    """

    if "events" in item:
        return item["events"]

    if "sequence" in item:
        return item["sequence"]

    if "data" in item:
        return item["data"]

    raise KeyError(
        f"Could not find event sequence in sample keys: "
        f"{list(item.keys())}"
    )


def extract_label(item):
    """
    Adapt if your dataset uses another label field.
    """

    if "label" in item:
        return int(item["label"])

    if "target" in item:
        return int(item["target"])

    raise KeyError(
        f"Could not find label in sample keys: "
        f"{list(item.keys())}"
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = load_test_set(TEST_PATH)

    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]

    print("=" * 70)
    print("SOCIA -- LLM BASELINE EVALUATION")
    print("=" * 70)

    print(f"Model       : {MODEL}")
    print(f"Test set    : {TEST_PATH}")
    print(f"Samples     : {len(samples)}")
    print()

    y_true = []
    y_pred = []
    y_score = []

    detailed_results = []

    start_time = time.time()

    for i, item in enumerate(samples):

        events = extract_events(item)
        true_label = extract_label(item)

        try:
            pred_label, confidence, raw = call_ollama(events)

        except Exception as e:
            print(
                f"[ERROR] sample {i}: {e}"
            )

            detailed_results.append({
                "index": i,
                "true_label": true_label,
                "prediction": None,
                "confidence": None,
                "error": str(e),
            })

            continue

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_score.append(confidence)

        detailed_results.append({
            "index": i,
            "true_label": true_label,
            "prediction": pred_label,
            "confidence": confidence,
            "raw_response": raw,
        })

        if (i + 1) % 50 == 0:
            print(
                f"Processed {i + 1}/{len(samples)}"
            )

    elapsed = time.time() - start_time

    # ========================================================
    # METRICS
    # ========================================================

    if not y_true:
        raise RuntimeError(
            "No successful LLM predictions."
        )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            y_score,
        )
    except ValueError:
        roc_auc = None

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "model": MODEL,
        "test_set": str(TEST_PATH),
        "n_requested": len(samples),
        "n_evaluated": len(y_true),
        "n_failed": len(samples) - len(y_true),

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,

        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },

        "runtime_seconds": elapsed,
        "seconds_per_sample": elapsed / len(y_true),

        "predictions": detailed_results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Evaluated samples : {len(y_true)}")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1                : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC           : {roc_auc:.4f}")
    else:
        print("ROC-AUC           : N/A")

    print()
    print("Confusion matrix:")
    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TP = {tp}")

    print()
    print(f"Runtime            : {elapsed:.2f}s")
    print(
        f"Per sample         : "
        f"{elapsed / len(y_true):.2f}s"
    )

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()