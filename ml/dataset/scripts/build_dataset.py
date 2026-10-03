"""
Dataset V2 - full pipeline.

    pattern pool (category-set + tuple disjoint across splits)
        -> per-split user/sample allocation
        -> direct construction of each SEQUENCE_LENGTH-event sample
           (positive or one of six negative subtypes)
        -> write train/val/test files + eval metadata + config snapshot
           + generation report

V1 is never read or written by this script. Output goes to OUT_DIR below
(research/dataset/v2/data by default), never research/dataset/v1/data.

Run:
    python build_dataset.py
"""
import json
import os
import numpy as np
from typing import List, Dict, Any

from config import (
    BASE_SEED, TARGET_TOTAL_PER_SPLIT, POSITIVE_SHARE,
    NEGATIVE_SUBTYPE_SHARES, SAMPLES_PER_USER_RANGE, SEQUENCE_LENGTH,
    N_COUNTERFACTUAL_BASE_SAMPLES,
)
from generator import build_pattern_pool, pools_by_split, verify_pool_disjointness
from simulate_user import build_sample, SUBTYPES
from counterfactuals import build_counterfactual_set

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "data"))


def _subtype_plan(total: int) -> Dict[str, int]:
    """Exact integer counts per subtype for one split, summing to `total`."""
    n_pos = int(round(total * POSITIVE_SHARE))
    n_neg_total = total - n_pos
    plan = {"positive": n_pos}
    remaining = n_neg_total
    items = list(NEGATIVE_SUBTYPE_SHARES.items())
    for name, share in items[:-1]:
        n = int(round(n_neg_total * share))
        plan[name] = n
        remaining -= n
    plan[items[-1][0]] = remaining  # last subtype absorbs rounding remainder
    assert sum(plan.values()) == total
    return plan


def _make_user_stream(split: str, seed: int):
    """Yields (user_id) values; every user belongs to exactly this split by
    construction (split-namespaced id), never assigned post hoc."""
    rng = np.random.default_rng(seed)
    idx = 0
    while True:
        idx += 1
        yield f"{split}_user_{idx:05d}", rng


def build_split(split: str, patterns_this_split, total: int, seed: int):
    plan = _subtype_plan(total)
    order = []
    for subtype, n in plan.items():
        order += [subtype] * n
    rng_shuffle = np.random.default_rng(seed + 999)
    rng_shuffle.shuffle(order)

    rows = []
    user_idx = 0
    user_rng = np.random.default_rng(seed + 7)
    remaining_for_user = 0
    current_user = None

    for i, subtype in enumerate(order):
        if remaining_for_user <= 0:
            user_idx += 1
            current_user = f"{split}_user_{user_idx:05d}"
            remaining_for_user = int(user_rng.integers(
                SAMPLES_PER_USER_RANGE[0], SAMPLES_PER_USER_RANGE[1] + 1))
        remaining_for_user -= 1

        pattern = patterns_this_split[i % len(patterns_this_split)]
        # rotate deterministically through the split's own pattern pool;
        # additionally reshuffle pattern order per-sample via the seed so
        # adjacent samples don't always share a pattern.
        pattern_pick_rng = np.random.default_rng(seed + 1000 + i)
        pattern = patterns_this_split[int(pattern_pick_rng.integers(0, len(patterns_this_split)))]

        # Deterministic per-sample seed from (split, index) only - no
        # Python hash() (unseeded/process-randomized for strings), no
        # collisions across splits (each split has its own offset band).
        sample_seed = (BASE_SEED * 1_000_003 + i
                        + {"train": 0, "val": 1, "test": 2}[split] * 10_000_000) % (2**31 - 1)
        rng = np.random.default_rng(sample_seed)

        sample_id = f"{split}_{i:06d}"
        row = build_sample(rng, subtype, pattern, sample_id, current_user, split, sample_seed)
        rows.append(row)

    return rows, plan


def _write_jsonl(path, dicts):
    with open(path, "w", encoding="utf-8") as f:
        for d in dicts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    patterns = build_pattern_pool(seed=BASE_SEED)
    pool_report = verify_pool_disjointness(patterns)
    print("Pattern pool disjointness:", json.dumps(pool_report))
    if not pool_report["all_clear"]:
        raise RuntimeError("Pattern pool is not split-disjoint; refusing to generate.")

    by_split_patterns = pools_by_split(patterns)

    all_rows = {}
    plans = {}
    for split in ["train", "val", "test"]:
        total = TARGET_TOTAL_PER_SPLIT[split]
        seed = BASE_SEED + {"train": 0, "val": 1, "test": 2}[split] * 100_000
        rows, plan = build_split(split, by_split_patterns[split], total, seed)
        all_rows[split] = rows
        plans[split] = plan
        label_counts = {0: 0, 1: 0}
        for r in rows:
            label_counts[r["training_example"]["label"]] += 1
        print(f"[{split}] total={len(rows)} plan={plan} label_counts={label_counts}")

    # ---------------- write per-split files ----------------
    for split in ["train", "val", "test"]:
        _write_jsonl(os.path.join(OUT_DIR, f"{split}.jsonl"),
                     [r["training_example"] for r in all_rows[split]])
        _write_jsonl(os.path.join(OUT_DIR, f"{split}_eval_metadata.jsonl"),
                     [r["eval_metadata"] for r in all_rows[split]])

    # ---------------- pattern pool ----------------
    with open(os.path.join(OUT_DIR, "pattern_pool.json"), "w", encoding="utf-8") as f:
        json.dump([
            {"pattern_id": p.pattern_id, "categories": list(p.tuple_categories),
             "k": p.k, "split": p.split}
            for p in patterns
        ], f, indent=2, ensure_ascii=False)

    # ---------------- user-disjointness check (should be trivially true) ----------------
    users_by_split = {
        s: set(r["training_example"]["user_id"] for r in all_rows[s])
        for s in ["train", "val", "test"]
    }
    user_overlap = {
        f"{a}_{b}": len(users_by_split[a] & users_by_split[b])
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    }

    # ---------------- counterfactual eval set (test split only) ----------------
    cf_rows = build_counterfactual_set(
        all_rows["test"], by_split_patterns["test"],
        n_base=N_COUNTERFACTUAL_BASE_SAMPLES, base_seed=BASE_SEED + 555_555,
    )
    _write_jsonl(os.path.join(OUT_DIR, "counterfactual_eval.jsonl"), cf_rows)

    # ---------------- generation report ----------------
    report = {
        "sequence_length": SEQUENCE_LENGTH,
        "base_seed": BASE_SEED,
        "pattern_pool_disjointness": pool_report,
        "target_total_per_split": TARGET_TOTAL_PER_SPLIT,
        "subtype_plan_per_split": plans,
        "user_overlap_across_splits": user_overlap,
        "n_users_per_split": {s: len(v) for s, v in users_by_split.items()},
        "n_counterfactual_rows": len(cf_rows),
    }
    with open(os.path.join(OUT_DIR, "generation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nGeneration complete.")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
