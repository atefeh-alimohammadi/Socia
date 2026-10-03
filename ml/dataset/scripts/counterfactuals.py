"""
Dataset V2 - counterfactual evaluation set (test split only).

For a subset of genuine TEST positives, produce controlled, single-factor
transforms of the same base sample. Each transformed sample links back to
its base via `base_sample_id` + shares a `counterfactual_group_id`. This is
meant to be run through the same model as ordinary test samples, scoring
how much the prediction changes under each controlled intervention - this
connects directly to the existing permutation/inconsistency diagnostics.

Transforms (each expected, under a correctly-behaving recurrence-sensitive
model, to push the prediction toward NEGATIVE relative to the base):

  * order_permutation      - permute each site's within-burst order away
                              from canonical order (composition + timing
                              untouched).
  * timestamp_collapse     - collapse the inter-site gaps to below the
                              persistence threshold (identity + order
                              untouched, only timing changes).
  * identity_substitution  - replace ONE site's tuple with a fresh random
                              tuple, breaking cross-site identity
                              consistency (only that site's categories
                              change; timing untouched).

This module never writes into a training file; `counterfactual_eval.jsonl`
is evaluation-only, exactly like V1's raw_events_debug.jsonl was.
"""
import copy
import zlib
import numpy as np
from typing import List, Dict, Any


def _stable_seed(*parts) -> int:
    """Deterministic seed from strings, independent of PYTHONHASHSEED.
    Python's built-in hash() is process-randomized for str/bytes and must
    never be used as a generation seed (this is exactly the class of
    hazard check_reproducibility.py flags in V1)."""
    s = "||".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(s) & 0x7FFFFFFF

from config import COUNTERFACTUAL_TRANSFORMS, CATEGORIES
from simulate_user import (
    _non_canonical_permutation, _fresh_random_tuple, _min_gap_days,
)


def _events_from_groups(base_events, base_group_positions, new_group_categories=None,
                         new_group_time_shift_days=None):
    """
    Rebuild the event list from a base sample's events, given per-group
    (per-site) overrides. Only touches events at recorded group positions;
    everything else (background) is left untouched.
    """
    events = copy.deepcopy(base_events)
    for gi, positions in enumerate(base_group_positions):
        if not positions:
            continue
        if new_group_categories is not None and new_group_categories[gi] is not None:
            for p, cat in zip(positions, new_group_categories[gi]):
                events[p]["category"] = cat
                events[p]["category_id"] = CATEGORIES.index(cat)
        if new_group_time_shift_days is not None and new_group_time_shift_days[gi] != 0.0:
            shift_h = new_group_time_shift_days[gi] * 24.0
            for p in positions:
                events[p]["delta_hours"] = round(events[p]["delta_hours"] + shift_h, 3)
    events.sort(key=lambda e: e["delta_hours"])
    return events


def _apply_order_permutation(rng, base_meta, base_events):
    groups = base_meta["group_global_positions"]
    site_cats = base_meta.get("site_categories")
    if not site_cats or len(groups) < 1:
        return None
    new_cats = []
    for cats in site_cats:
        new_cats.append(list(_non_canonical_permutation(rng, cats)))
    events = _events_from_groups(base_events, groups, new_group_categories=new_cats)
    return events


def _apply_timestamp_collapse(rng, base_meta, base_events):
    groups = base_meta["group_global_positions"]
    if len(groups) < 2:
        return None
    min_gap, _ = _min_gap_days()
    # shift every site after the first inward, toward the first site,
    # so consecutive gaps fall well below the persistence threshold.
    shifts = [0.0]
    site_start_order = list(range(len(groups)))
    for i in site_start_order[1:]:
        shifts.append(-(min_gap * 0.9) * i)  # progressively pulled back
    events = _events_from_groups(base_events, groups, new_group_time_shift_days=shifts)
    return events


def _apply_identity_substitution(rng, base_meta, base_events):
    groups = base_meta["group_global_positions"]
    site_cats = base_meta.get("site_categories")
    if not site_cats or len(groups) < 2:
        return None
    victim = int(rng.integers(0, len(groups)))
    new_cats = [None] * len(groups)
    fresh = _fresh_random_tuple(rng, len(site_cats[victim]), exclude_tuple=site_cats[victim])
    new_cats[victim] = list(fresh)
    events = _events_from_groups(base_events, groups, new_group_categories=new_cats)
    return events


_APPLY = {
    "order_permutation": _apply_order_permutation,
    "timestamp_collapse": _apply_timestamp_collapse,
    "identity_substitution": _apply_identity_substitution,
}


def build_counterfactual_set(test_rows: List[Dict[str, Any]], test_patterns, n_base: int, base_seed: int):
    rng = np.random.default_rng(base_seed)
    positives = [r for r in test_rows if r["training_example"]["label"] == 1
                 and r["eval_metadata"].get("n_occ_intended", 0) >= 2]
    n_base = min(n_base, len(positives))
    idx = rng.choice(len(positives), size=n_base, replace=False)
    base_sample_list = [positives[i] for i in idx]

    out = []
    for base in base_sample_list:
        te, meta = base["training_example"], base["eval_metadata"]
        group_id = f"cf_{te['sample_id']}"
        out.append({
            "counterfactual_group_id": group_id,
            "role": "base",
            "transform": None,
            "base_sample_id": te["sample_id"],
            "sample_id": te["sample_id"],
            "user_id": te["user_id"],
            "events": te["events"],
            "label": te["label"],
        })
        for transform in COUNTERFACTUAL_TRANSFORMS:
            fn = _APPLY[transform]
            local_rng = np.random.default_rng(_stable_seed(group_id, transform))
            new_events = fn(local_rng, meta, te["events"])
            if new_events is None:
                continue
            out.append({
                "counterfactual_group_id": group_id,
                "role": "transformed",
                "transform": transform,
                "base_sample_id": te["sample_id"],
                "sample_id": f"{te['sample_id']}__{transform}",
                "user_id": te["user_id"],
                "events": new_events,
                # Expected label under the transform, NOT a claim about
                # what any specific model will predict - for scoring
                # prediction-shift, not for training.
                "expected_label": 0,
            })
    return out
