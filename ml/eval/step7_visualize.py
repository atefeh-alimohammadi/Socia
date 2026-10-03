#!/usr/bin/env python3
"""
Step 7 — Visualization.

Reads ONLY the saved outputs of steps 2-6 (final_evaluation_report.json,
subtype_analysis.json, counterfactual_analysis.json,
recurrence_analysis.json, error_analysis.json). Does NOT re-run model
inference and does NOT recompute any metric — every number plotted here
is read verbatim from those files.

Usage:
    python step7_visualize.py --out-dir <out_dir>

Writes PNGs to <out_dir>/figures/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ["baseline_timing_category", "v3a", "v3c"]
MODEL_LABELS = {"baseline_timing_category": "Baseline\n(timing+category)", "v3a": "V3-A", "v3c": "V3-C"}
MODEL_COLORS = {"baseline_timing_category": "#9aa0a6", "v3a": "#4285f4", "v3c": "#ea4335"}

SUBTYPE_ORDER = ["timing_matched", "order_permutation", "boundary_single_occurrence",
                  "category_identity", "boundary_tight_burst", "pure_background"]
TRANSFORM_ORDER = ["order_permutation", "timestamp_collapse", "identity_substitution"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\AI Companion\Socia\ml\dataset\scripts\data"),
        help="Directory containing test_eval_metadata.jsonl",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"D:\AI Companion\Socia\ml\eval\eval_outputs"),
        help="Directory containing Step 2 prediction outputs and where Step 3 outputs will be written",
    )
    return p.parse_args()

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fig1_overall_comparison(out, fig_dir):
    report = load(out / "final_evaluation_report.json")["models"]
    metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    metric_labels = ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall"]

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model_key in enumerate(MODEL_ORDER):
        vals = [report[model_key][m] for m in metrics]
        ax.bar(x + (i - 1) * width, vals, width, label=MODEL_LABELS[model_key], color=MODEL_COLORS[model_key])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Test-set performance: Baseline vs V3-A vs V3-C\n(threshold selected on validation only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_overall_comparison.png", dpi=150)
    plt.close(fig)


def fig2_subtype_fp_rate(out, fig_dir):
    rows = load(out / "subtype_analysis.json")
    by_model_subtype = {(r["model"], r["subtype"]): r for r in rows if r["subtype"] != "positive"}

    x = np.arange(len(SUBTYPE_ORDER))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, model_key in enumerate(MODEL_ORDER):
        vals = [by_model_subtype.get((model_key, s), {}).get("false_positive_rate", np.nan) for s in SUBTYPE_ORDER]
        ax.bar(x + (i - 1) * width, vals, width, label=MODEL_LABELS[model_key], color=MODEL_COLORS[model_key])
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in SUBTYPE_ORDER], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("False-positive rate")
    ax.set_title("False-positive rate by negative subtype (test set)")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_subtype_fp_rate.png", dpi=150)
    plt.close(fig)


def fig3_counterfactual_shift(out, fig_dir):
    rows = load(out / "counterfactual_analysis.json")
    models_to_plot = ["v3a", "v3c"]
    by = {(r["model"], r["transform"]): r for r in rows}

    x = np.arange(len(TRANSFORM_ORDER))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for i, model_key in enumerate(models_to_plot):
        orig = [by[(model_key, t)]["mean_original_prob"] for t in TRANSFORM_ORDER]
        cf = [by[(model_key, t)]["mean_counterfactual_prob"] for t in TRANSFORM_ORDER]
        off = (i - 0.5) * width
        ax.bar(x + off - width / 4, orig, width / 2, label=f"{model_key} original", color=MODEL_COLORS[model_key], alpha=0.5)
        ax.bar(x + off + width / 4, cf, width / 2, label=f"{model_key} transformed", color=MODEL_COLORS[model_key])
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in TRANSFORM_ORDER], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean predicted probability")
    ax.set_title("Original vs. counterfactual probability")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for i, model_key in enumerate(models_to_plot):
        change = [by[(model_key, t)]["mean_change"] for t in TRANSFORM_ORDER]
        ax.bar(x + (i - 0.5) * width, change, width, label=MODEL_LABELS[model_key], color=MODEL_COLORS[model_key])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in TRANSFORM_ORDER], fontsize=9)
    ax.set_ylabel("Mean probability change (transformed - original)")
    ax.set_title("Counterfactual probability shift\n(expected: negative)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_counterfactual_shift.png", dpi=150)
    plt.close(fig)


def fig4_recurrence_probability(out, fig_dir):
    rows = load(out / "recurrence_analysis.json")
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_key in MODEL_ORDER:
        sub = sorted([r for r in rows if r["model"] == model_key], key=lambda r: r["n_occ_intended"])
        xs = [r["n_occ_intended"] for r in sub]
        ys = [r["mean_pred_prob"] for r in sub]
        ax.plot(xs, ys, marker="o", label=MODEL_LABELS[model_key].replace("\n", " "), color=MODEL_COLORS[model_key])
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("n_occ_intended (recurrence count)")
    ax.set_ylabel("Mean predicted probability (positives only)")
    ax.set_title("Predicted probability vs. recurrence count")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_recurrence_probability.png", dpi=150)
    plt.close(fig)


def fig5_error_breakdown(out, fig_dir):
    err = load(out / "error_analysis.json")
    models_to_plot = ["v3a", "v3c"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(SUBTYPE_ORDER))
    width = 0.35
    for i, model_key in enumerate(models_to_plot):
        shares = err[model_key]["fp_share_of_all_fps_by_negative_subtype"]
        vals = [shares.get(s, 0.0) for s in SUBTYPE_ORDER]
        ax.bar(x + (i - 0.5) * width, vals, width, label=MODEL_LABELS[model_key], color=MODEL_COLORS[model_key])
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in SUBTYPE_ORDER], fontsize=9)
    ax.set_ylabel("Share of all false positives")
    ax.set_title("Where false positives come from (test set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_error_breakdown.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out = args.out_dir
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig1_overall_comparison(out, fig_dir)
    fig2_subtype_fp_rate(out, fig_dir)
    fig3_counterfactual_shift(out, fig_dir)
    fig4_recurrence_probability(out, fig_dir)
    fig5_error_breakdown(out, fig_dir)

    print(f"Figures written to: {fig_dir}")
    for p in sorted(fig_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
