"""
Step 7 — Visualization.

Reads ONLY saved outputs from Steps 2-6:

    - final_evaluation_report.json
    - subtype_analysis.json
    - counterfactual_analysis.json
    - recurrence_analysis.json
    - error_analysis.json

Does NOT run model inference.
Does NOT recompute model metrics.

Every plotted value is read from the saved evaluation outputs.

Usage:
    python -m ml.eval.step7_visualize

Optional:
    python -m ml.eval.step7_visualize --out-dir <out_dir>

Writes PNGs to:
    <out_dir>/figures/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_OUT_DIR = (
    SCRIPT_DIR / "eval_outputs"
)


# ============================================================================
# Model configuration
# ============================================================================

BENCHMARK_MODEL_ORDER = [
    "baseline_timing_category",
    "v2",
    "v3a",
    "v3c",
    "v3a_bucketed_recurrence",
]


SUBTYPE_MODEL_ORDER = [
    "baseline_timing_category",
    "v3a",
    "v3c",
]


MODEL_LABELS = {
    "baseline_timing_category": "Baseline\n(timing+category)",
    "v2": "V2",
    "v3a": "V3-A",
    "v3c": "V3-C",
    "v3a_bucketed_recurrence": "V3-A\nBucketed",
}


MODEL_COLORS = {
    "baseline_timing_category": "#9aa0a6",
    "v2": "#34a853",
    "v3a": "#4285f4",
    "v3c": "#ea4335",
    "v3a_bucketed_recurrence": "#ab47bc",
}


SUBTYPE_ORDER = [
    "timing_matched",
    "order_permutation",
    "boundary_single_occurrence",
    "category_identity",
    "boundary_tight_burst",
    "pure_background",
]


TRANSFORM_ORDER = [
    "order_permutation",
    "timestamp_collapse",
    "identity_substitution",
]


# ============================================================================
# Arguments / loading
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            "Directory containing Step 2-6 evaluation outputs "
            "and where figures will be written."
        ),
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required Step output not found: {path}"
        )


def load_json(path: Path) -> Any:
    require_file(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def require_finite(
    value: Any,
    description: str,
) -> float:

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric value for {description}: {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"Non-finite value for {description}: {value!r}"
        )

    return result


# ============================================================================
# Small schema adapters
# ============================================================================

def unwrap_models(payload: Any) -> Dict[str, Any]:
    """
    Return the model mapping from either:

        {"models": {...}}

    or directly:

        {...}
    """

    if isinstance(payload, dict):
        if isinstance(payload.get("models"), dict):
            return payload["models"]

        return payload

    raise ValueError(
        "Expected a JSON object containing model results."
    )


def unwrap_rows(payload: Any) -> List[Dict[str, Any]]:
    """
    Accept a list directly or common wrapper forms such as:

        {"rows": [...]}
        {"results": [...]}
        {"analysis": [...]}
    """

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in (
            "rows",
            "results",
            "analysis",
            "subtypes",
            "transforms",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                return value

    raise ValueError(
        "Expected a JSON list or an object containing a result list."
    )



def get_metric(model_entry, metric_name, split="test"):


    if not isinstance(model_entry, dict):
        raise TypeError(
            f"Expected model entry to be a dict, got {type(model_entry).__name__}"
        )

    split_data = model_entry.get(split)

    if isinstance(split_data, dict) and metric_name in split_data:
        value = split_data[metric_name]

        if value is None:
            raise KeyError(
                f"Metric '{metric_name}' is present under '{split}' "
                f"but has value None."
            )

        return float(value)

    available_splits = [
        key for key, value in model_entry.items()
        if isinstance(value, dict)
    ]

    raise KeyError(
        f"Metric '{metric_name}' not found under split '{split}'. "
        f"Available model-level keys: {list(model_entry.keys())}. "
        f"Available split-like keys: {available_splits}"
    )



# ============================================================================
# Figure 1 — overall benchmark comparison
# ============================================================================

def fig1_overall_comparison(
    out: Path,
    fig_dir: Path,
) -> None:

    report_payload = load_json(
        out / "final_evaluation_report.json"
    )

    report = unwrap_models(report_payload)

    metrics = [
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
    ]

    metric_labels = [
        "ROC-AUC",
        "PR-AUC",
        "F1",
        "Precision",
        "Recall",
    ]

    for model_key in BENCHMARK_MODEL_ORDER:
        if model_key not in report:
            raise ValueError(
                f"Model {model_key!r} missing from "
                "final_evaluation_report.json."
            )

        for metric in metrics:
            get_metric(
                report[model_key],
                metric,
            )

    x = np.arange(len(metrics))

    n_models = len(
        BENCHMARK_MODEL_ORDER
    )

    width = 0.15

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    for i, model_key in enumerate(
        BENCHMARK_MODEL_ORDER
    ):

        values = [
            get_metric(
                report[model_key],
                metric,
            )
            for metric in metrics
        ]

        offset = (
            i - (n_models - 1) / 2
        ) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=MODEL_LABELS[model_key],
            color=MODEL_COLORS[model_key],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Test-set performance across benchmark models\n"
        "(threshold selected on validation only)"
    )

    ax.legend(
        fontsize=9
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        fig_dir / "fig1_overall_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# Figure 2 — false-positive rate by negative subtype
# ============================================================================

def _extract_subtype_rows(
    payload: Any,
) -> List[Dict[str, Any]]:
    """
    Normalize Step 3 output.

    The expected conceptual fields are:

        model
        subtype
        false_positive_rate

    The function accepts either a list or a wrapped list.
    """

    rows = unwrap_rows(payload)

    normalized = []

    for row in rows:

        if not isinstance(row, dict):
            continue

        if (
            "model" not in row
            or "subtype" not in row
        ):
            continue

        normalized.append(row)

    if not normalized:
        raise ValueError(
            "Could not find model/subtype rows in "
            "subtype_analysis.json."
        )

    return normalized


def fig2_subtype_fp_rate(
    out: Path,
    fig_dir: Path,
) -> None:

    payload = load_json(
        out / "subtype_analysis.json"
    )

    rows = _extract_subtype_rows(
        payload
    )

    by_model_subtype = {}

    for row in rows:

        model = str(row["model"])
        subtype = str(row["subtype"])

        if subtype == "positive":
            continue

        if "false_positive_rate" not in row:
            raise ValueError(
                "subtype_analysis.json row is missing "
                "'false_positive_rate'."
            )

        by_model_subtype[
            (model, subtype)
        ] = require_finite(
            row["false_positive_rate"],
            f"{model}/{subtype} false_positive_rate",
        )

    x = np.arange(
        len(SUBTYPE_ORDER)
    )

    n_models = len(
        SUBTYPE_MODEL_ORDER
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    for i, model_key in enumerate(
        SUBTYPE_MODEL_ORDER
    ):

        values = []

        for subtype in SUBTYPE_ORDER:

            values.append(
                by_model_subtype.get(
                    (model_key, subtype),
                    np.nan,
                )
            )

        offset = (
            i - (n_models - 1) / 2
        ) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=MODEL_LABELS[model_key],
            color=MODEL_COLORS[model_key],
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            subtype.replace(
                "_",
                "\n",
            )
            for subtype in SUBTYPE_ORDER
        ],
        fontsize=9,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        "False-positive rate"
    )

    ax.set_title(
        "False-positive rate by negative subtype (test set)"
    )

    ax.axhline(
        0.5,
        color="gray",
        linewidth=0.8,
        linestyle=":",
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        fig_dir / "fig2_subtype_fp_rate.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# Figure 3 — counterfactual probability shift
# ============================================================================

def _extract_counterfactual_rows(
    payload: Any,
) -> List[Dict[str, Any]]:
    """
    Normalize Step 4 output.

    Expected conceptual fields:

        model
        transform
        mean_original_prob
        mean_counterfactual_prob
        mean_change
    """

    rows = unwrap_rows(payload)

    normalized = []

    for row in rows:

        if not isinstance(row, dict):
            continue

        if (
            "model" in row
            and "transform" in row
        ):
            normalized.append(row)

    if not normalized:
        raise ValueError(
            "Could not find model/transform rows in "
            "counterfactual_analysis.json."
        )

    return normalized


def _get_counterfactual_metric(
    row: Dict[str, Any],
    primary: str,
    alternatives: List[str],
    description: str,
) -> float:

    if primary in row:
        return require_finite(
            row[primary],
            description,
        )

    for key in alternatives:
        if key in row:
            return require_finite(
                row[key],
                description,
            )

    raise KeyError(
        f"Could not find {description}. "
        f"Available keys: {sorted(row.keys())}"
    )


def fig3_counterfactual_shift(
    out: Path,
    fig_dir: Path,
) -> None:

    payload = load_json(
        out / "counterfactual_analysis.json"
    )

    rows = _extract_counterfactual_rows(
        payload
    )

    by = {}

    for row in rows:

        model = str(row["model"])
        transform = str(row["transform"])

        by[
            (model, transform)
        ] = row

    models_to_plot = [
        "v3a",
        "v3c",
    ]

    x = np.arange(
        len(TRANSFORM_ORDER)
    )

    width = 0.35

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    # ------------------------------------------------------------------
    # Original vs transformed
    # ------------------------------------------------------------------

    ax = axes[0]

    for i, model_key in enumerate(
        models_to_plot
    ):

        original = []
        counterfactual = []

        for transform in TRANSFORM_ORDER:

            key = (
                model_key,
                transform,
            )

            if key not in by:
                raise ValueError(
                    f"Missing counterfactual result for "
                    f"{model_key}/{transform}."
                )

            row = by[key]

            original.append(
                _get_counterfactual_metric(
                    row,
                    "mean_original_prob",
                    ["mean_original_probability"],
                    "mean original probability",
                )
            )

            counterfactual.append(
                _get_counterfactual_metric(
                    row,
                    "mean_counterfactual_prob",
                    ["mean_counterfactual_probability"],
                    "mean counterfactual probability",
                )
            )

        off = (
            i - 0.5
        ) * width

        label = MODEL_LABELS[
            model_key
        ].replace(
            "\n",
            " ",
        )

        ax.bar(
            x + off - width / 4,
            original,
            width / 2,
            label=f"{label} original",
            color=MODEL_COLORS[model_key],
            alpha=0.5,
        )

        ax.bar(
            x + off + width / 4,
            counterfactual,
            width / 2,
            label=f"{label} transformed",
            color=MODEL_COLORS[model_key],
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            transform.replace(
                "_",
                "\n",
            )
            for transform in TRANSFORM_ORDER
        ],
        fontsize=9,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        "Mean predicted probability"
    )

    ax.set_title(
        "Original vs. counterfactual probability"
    )

    ax.legend(
        fontsize=8
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    # ------------------------------------------------------------------
    # Probability change
    # ------------------------------------------------------------------

    ax = axes[1]

    for i, model_key in enumerate(
        models_to_plot
    ):

        changes = []

        for transform in TRANSFORM_ORDER:

            key = (
                model_key,
                transform,
            )

            row = by[key]

            changes.append(
                _get_counterfactual_metric(
                    row,
                    "mean_change",
                    ["mean_probability_change"],
                    "mean probability change",
                )
            )

        ax.bar(
            x + (
                i - 0.5
            ) * width,
            changes,
            width,
            label=MODEL_LABELS[
                model_key
            ],
            color=MODEL_COLORS[model_key],
        )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            transform.replace(
                "_",
                "\n",
            )
            for transform in TRANSFORM_ORDER
        ],
        fontsize=9,
    )

    ax.set_ylabel(
        "Mean probability change\n"
        "(transformed - original)"
    )

    ax.set_title(
        "Counterfactual probability shift\n"
        "(expected: negative)"
    )

    ax.legend(
        fontsize=8
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        fig_dir / "fig3_counterfactual_shift.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# Figure 4 — recurrence probability
# ============================================================================

def _extract_recurrence_models(
    payload: Any,
) -> Dict[str, Any]:
    """
    Step 5 current output:

        {
            "metadata": {...},
            "models": {
                "baseline_timing_category": {
                    "threshold": ...,
                    "buckets": {
                        "2": {
                            "n": ...,
                            "mean_probability": ...,
                            "median_probability": ...,
                            "recall_at_threshold": ...
                        }
                    }
                }
            }
        }
    """

    if not isinstance(payload, dict):
        raise ValueError(
            "recurrence_analysis.json must be an object."
        )

    models = payload.get(
        "models"
    )

    if not isinstance(models, dict):
        raise ValueError(
            "recurrence_analysis.json does not contain "
            "a valid 'models' object."
        )

    return models


def fig4_recurrence_probability(
    out: Path,
    fig_dir: Path,
) -> None:

    payload = load_json(
        out / "recurrence_analysis.json"
    )

    models = _extract_recurrence_models(
        payload
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    plotted_any = False

    for model_key in SUBTYPE_MODEL_ORDER:

        if model_key not in models:
            raise ValueError(
                f"Model {model_key!r} missing from "
                "recurrence_analysis.json."
            )

        model_info = models[
            model_key
        ]

        buckets = model_info.get(
            "buckets"
        )

        if not isinstance(buckets, dict):
            raise ValueError(
                f"Invalid recurrence buckets for "
                f"{model_key}."
            )

        bucket_rows = []

        for bucket_key, bucket_info in buckets.items():

            try:
                n_occ = int(bucket_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid recurrence bucket {bucket_key!r} "
                    f"for {model_key}."
                ) from exc

            if "mean_probability" not in bucket_info:
                raise ValueError(
                    f"Missing mean_probability for "
                    f"{model_key}, bucket {bucket_key}."
                )

            probability = require_finite(
                bucket_info["mean_probability"],
                f"{model_key}/{bucket_key} mean_probability",
            )

            bucket_rows.append(
                (
                    n_occ,
                    probability,
                )
            )

        bucket_rows.sort(
            key=lambda x: x[0]
        )

        if not bucket_rows:
            continue

        xs = [
            row[0]
            for row in bucket_rows
        ]

        ys = [
            row[1]
            for row in bucket_rows
        ]

        ax.plot(
            xs,
            ys,
            marker="o",
            label=MODEL_LABELS[
                model_key
            ].replace(
                "\n",
                " ",
            ),
            color=MODEL_COLORS[
                model_key
            ],
        )

        plotted_any = True

    if not plotted_any:
        raise ValueError(
            "No recurrence data available for Figure 4."
        )

    ax.set_xlabel(
        "n_occ_intended (recurrence count)"
    )

    ax.set_ylabel(
        "Mean predicted probability\n"
        "(positives only)"
    )

    ax.set_title(
        "Predicted probability vs. recurrence count"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.legend()

    ax.grid(
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        fig_dir / "fig4_recurrence_probability.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# Figure 5 — error breakdown
# ============================================================================

def fig5_error_breakdown(
    out: Path,
    fig_dir: Path,
) -> None:

    payload = load_json(
        out / "error_analysis.json"
    )

    models = unwrap_models(
        payload
    )

    models_to_plot = [
        "v3a",
        "v3c",
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    x = np.arange(
        len(SUBTYPE_ORDER)
    )

    width = 0.35

    for i, model_key in enumerate(
        models_to_plot
    ):

        if model_key not in models:
            raise ValueError(
                f"Model {model_key!r} missing from "
                "error_analysis.json."
            )

        model_info = models[
            model_key
        ]

        shares = model_info.get(
            "fp_share_of_all_fps_by_negative_subtype"
        )

        if not isinstance(shares, dict):
            raise ValueError(
                f"Model {model_key!r} is missing "
                "'fp_share_of_all_fps_by_negative_subtype'."
            )

        values = [
            require_finite(
                shares.get(
                    subtype,
                    0.0,
                ),
                f"{model_key}/{subtype} FP share",
            )
            for subtype in SUBTYPE_ORDER
        ]

        ax.bar(
            x + (
                i - 0.5
            ) * width,
            values,
            width,
            label=MODEL_LABELS[
                model_key
            ],
            color=MODEL_COLORS[
                model_key
            ],
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            subtype.replace(
                "_",
                "\n",
            )
            for subtype in SUBTYPE_ORDER
        ],
        fontsize=9,
    )

    ax.set_ylabel(
        "Share of all false positives"
    )

    ax.set_title(
        "Where false positives come from (test set)"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        fig_dir / "fig5_error_breakdown.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# Figure 6 — confusion matrices
# ============================================================================


def _get_confusion_value(model_entry, metric_name, model_name=None, split="test"):

    if not isinstance(model_entry, dict):
        raise TypeError(
            f"Expected model report entry to be a dict, "
            f"got {type(model_entry).__name__}"
        )

    split_data = model_entry.get(split)

    if not isinstance(split_data, dict):
        raise KeyError(
            f"{model_name or 'model'} has no valid '{split}' section. "
            f"Available keys: {list(model_entry.keys())}"
        )

    if metric_name not in {"tn", "fp", "fn", "tp"}:
        raise ValueError(
            f"Invalid confusion metric '{metric_name}'. "
            f"Expected one of: tn, fp, fn, tp"
        )

    if metric_name not in split_data:
        raise KeyError(
            f"{metric_name} {model_name or 'model'} missing from "
            f"model report under split '{split}'. "
            f"Available keys: {list(split_data.keys())}"
        )

    return int(split_data[metric_name])





def fig6_confusion_matrices(
    out: Path,
    fig_dir: Path,
) -> None:

    payload = load_json(
        out / "final_evaluation_report.json"
    )

    report = unwrap_models(
        payload
    )

    labels = {
        "baseline_timing_category": "Baseline",
        "v2": "V2",
        "v3a": "V3-A",
        "v3c": "V3-C",
        "v3a_bucketed_recurrence":
            "V3-A Bucketed",
    }

    fig, axes = plt.subplots(
        1,
        len(BENCHMARK_MODEL_ORDER),
        figsize=(16, 3.5),
    )

    if len(BENCHMARK_MODEL_ORDER) == 1:
        axes = [axes]

    for ax, model_key in zip(
        axes,
        BENCHMARK_MODEL_ORDER,
    ):

        if model_key not in report:
            raise ValueError(
                f"Model {model_key!r} missing from "
                "final_evaluation_report.json."
            )

        model_info = report[
            model_key
        ]

        tn = _get_confusion_value(
            model_info,
            "tn",
            f"{model_key} TN",
        )

        fp = _get_confusion_value(
            model_info,
            "fp",
            f"{model_key} FP",
        )

        fn = _get_confusion_value(
            model_info,
            "fn",
            f"{model_key} FN",
        )

        tp = _get_confusion_value(
            model_info,
            "tp",
            f"{model_key} TP",
        )

        matrix = np.array(
            [
                [tn, fp],
                [fn, tp],
            ]
        )

        ax.imshow(
            matrix,
            cmap="Blues",
            vmin=0,
        )

        ax.set_title(
            labels[model_key],
            fontsize=10,
        )

        ax.set_xticks(
            [0, 1]
        )

        ax.set_yticks(
            [0, 1]
        )

        ax.set_xticklabels(
            [
                "Pred 0",
                "Pred 1",
            ]
        )

        ax.set_yticklabels(
            [
                "True 0",
                "True 1",
            ]
        )

        for row in range(2):
            for col in range(2):

                ax.text(
                    col,
                    row,
                    str(
                        matrix[
                            row,
                            col,
                        ]
                    ),
                    ha="center",
                    va="center",
                    fontsize=11,
                )

    fig.suptitle(
        "Confusion matrices — frozen V2 test set"
    )

    fig.tight_layout()

    fig.savefig(
        fig_dir / "fig6_confusion_matrices.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# Figure 7 — accuracy
# ============================================================================

def fig7_accuracy_comparison(
    out: Path,
    fig_dir: Path,
) -> None:

    payload = load_json(
        out / "final_evaluation_report.json"
    )

    report = unwrap_models(
        payload
    )

    labels = [
        "Baseline",
        "V2",
        "V3-A",
        "V3-C",
        "V3-A Bucketed",
    ]

    values = []

    for model_key in BENCHMARK_MODEL_ORDER:

        if model_key not in report:
            raise ValueError(
                f"Model {model_key!r} missing from "
                "final_evaluation_report.json."
            )

        values.append(
            get_metric(
                report[model_key],
                "accuracy",
            )
        )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_title(
        "Accuracy comparison"
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        fig_dir / "fig7_accuracy_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    args = parse_args()

    out = args.out_dir.resolve()

    fig_dir = (
        out / "figures"
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("SOCIA — STEP 7: VISUALIZATION")
    print("=" * 78)
    print()
    print(
        f"Evaluation directory: {out}"
    )
    print(
        f"Figure directory    : {fig_dir}"
    )
    print()

    # -----------------------------------------------------------------
    # Generate all seven figures.
    #
    # No inference.
    # No metric recomputation.
    # Only saved Step 2-6 JSON outputs are read.
    # -----------------------------------------------------------------

    print("Generating Figure 1...")
    fig1_overall_comparison(
        out,
        fig_dir,
    )

    print("Generating Figure 2...")
    fig2_subtype_fp_rate(
        out,
        fig_dir,
    )

    print("Generating Figure 3...")
    fig3_counterfactual_shift(
        out,
        fig_dir,
    )

    print("Generating Figure 4...")
    fig4_recurrence_probability(
        out,
        fig_dir,
    )

    print("Generating Figure 5...")
    fig5_error_breakdown(
        out,
        fig_dir,
    )

    print("Generating Figure 6...")
    fig6_confusion_matrices(
        out,
        fig_dir,
    )

    print("Generating Figure 7...")
    fig7_accuracy_comparison(
        out,
        fig_dir,
    )

    print()
    print("=" * 78)
    print("STEP 7 COMPLETE")
    print("=" * 78)
    print()

    for path in sorted(
        fig_dir.glob("*.png")
    ):
        print(
            f"  {path.name}"
        )


if __name__ == "__main__":
    main()

