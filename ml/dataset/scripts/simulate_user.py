"""
Dataset V2 - per-sample construction ("simulate one synthetic user session").

Each sample is built DIRECTLY as a fixed-length (SEQUENCE_LENGTH) event
sequence, rather than sliced from one long shared per-user timeline the way
V1 did. This removes two V1 audit findings at the root:

  * ~97.7% of V1's selected windows overlapped another selected window of
    the same user (severe pseudo-replication from stride-1 sliding).
  * V1 drew negatives mostly from a separate 40%-of-users "control"
    population with P(label=1 | control user) == 0 by construction, making
    "is this a control user" itself a usable shortcut.

Every subtype below (the positive and all six negatives) is built through
the SAME `_sample_background` function and the SAME PATTERN_VOCAB_BIAS_PROB
policy, tied to a `reference_pattern_id` that every sample carries - so
background composition cannot itself correlate with the label.
"""
import numpy as np
from typing import List, Tuple, Dict, Any

from config import (
    CATEGORIES, SEQUENCE_LENGTH, WITHIN_OCC_GAP_HOURS_RANGE,
    BURST_SPAN_HOURS_CAP, N_OCC_CHOICES_POSITIVE, N_OCC_WEIGHTS_POSITIVE,
    MIN_INTER_OCC_GAP_DAYS_ABS, PERSISTENCE_RATIO,
    INTER_OCC_JITTER_LOGNORMAL_SIGMA, PATTERN_VOCAB_BIAS_PROB,
)

SUBTYPES = [
    "positive",
    "timing_matched",
    "category_identity",
    "order_permutation",
    "boundary_single_occurrence",
    "boundary_tight_burst",
    "pure_background",
]


# ---------------------------------------------------------------------------
# Building blocks shared by every subtype
# ---------------------------------------------------------------------------

def _site_events(rng, categories_in_order, start_day):
    """One burst: len(categories_in_order) events in the given order, each
    consecutive pair separated by WITHIN_OCC_GAP_HOURS_RANGE hours, capped
    so the whole site never exceeds BURST_SPAN_HOURS_CAP."""
    t = start_day
    out = [(categories_in_order[0], t)]
    for c in categories_in_order[1:]:
        gap_h = rng.uniform(*WITHIN_OCC_GAP_HOURS_RANGE)
        t_next = t + gap_h / 24.0
        if (t_next - start_day) * 24.0 > BURST_SPAN_HOURS_CAP:
            t_next = start_day + BURST_SPAN_HOURS_CAP / 24.0
        t = t_next
        out.append((c, t))
    return out


def _min_gap_days():
    burst_span_days = BURST_SPAN_HOURS_CAP / 24.0
    return max(MIN_INTER_OCC_GAP_DAYS_ABS, PERSISTENCE_RATIO * burst_span_days), burst_span_days


def _schedule_sites(rng, n_occ, gap_override_days=None):
    """
    Choose n_occ site start days such that every consecutive pair is
    separated by >= min_gap (persistence-satisfying), unless
    gap_override_days is given (used by boundary_tight_burst to violate
    persistence deliberately). Returns
    (start_days, total_span_days, min_gap_used, burst_span_days).
    """
    min_gap, burst_span_days = _min_gap_days()
    gap_to_use = min_gap if gap_override_days is None else gap_override_days
    mean_interval = max(gap_to_use, rng.uniform(gap_to_use, gap_to_use * 2.5))

    starts = [rng.uniform(0, mean_interval * 0.5)]
    for _ in range(n_occ - 1):
        jitter = np.exp(rng.normal(0, INTER_OCC_JITTER_LOGNORMAL_SIGMA))
        nxt = starts[-1] + max(mean_interval * jitter, gap_to_use)
        starts.append(nxt)
    total_span_days = starts[-1] + burst_span_days + rng.uniform(0.5, 3.0)
    return starts, total_span_days, min_gap, burst_span_days


def _sample_background(rng, n_slots, total_span_days, pattern_vocab):
    if n_slots <= 0:
        return []
    ts = rng.uniform(0, total_span_days, size=n_slots)
    out = []
    for t in ts:
        if pattern_vocab and rng.random() < PATTERN_VOCAB_BIAS_PROB:
            cat = rng.choice(pattern_vocab)
        else:
            cat = rng.choice(CATEGORIES)
        out.append((str(cat), float(t)))
    return out


def _fresh_random_tuple(rng, k, exclude_tuple=None):
    for _ in range(50):
        cats = tuple(rng.choice(CATEGORIES, size=k, replace=False).tolist())
        if exclude_tuple is None or cats != tuple(exclude_tuple):
            return cats
    return cats


def _non_canonical_permutation(rng, tup):
    tup = list(tup)
    if len(tup) < 2:
        return tuple(tup)
    for _ in range(20):
        perm = tup[:]
        rng.shuffle(perm)
        if perm != tup:
            return tuple(perm)
    return tuple(perm)


def _assemble(structured_groups: List[List[Tuple[str, float]]], background_events: List[Tuple[str, float]]):
    """
    structured_groups: list of "groups" (sites, or ungrouped singleton
    lists for category_identity), each a list of (category, day).
    Returns (sorted_events_for_model, group_global_positions).
    """
    tagged = []
    for gi, grp in enumerate(structured_groups):
        for (c, t) in grp:
            tagged.append((t, c, gi))
    for (c, t) in background_events:
        tagged.append((t, c, None))
    tagged.sort(key=lambda r: r[0])

    t0 = tagged[0][0]
    sorted_events = []
    group_positions = {i: [] for i in range(len(structured_groups))}
    for pos, (t, c, gi) in enumerate(tagged):
        sorted_events.append({
            "category_id": CATEGORIES.index(c),
            "category": c,
            "delta_hours": round((t - t0) * 24.0, 3),
        })
        if gi is not None:
            group_positions[gi].append(pos)
    return sorted_events, [group_positions[i] for i in range(len(structured_groups))]


def _pick_n_occ_positive(rng):
    return int(rng.choice(N_OCC_CHOICES_POSITIVE, p=N_OCC_WEIGHTS_POSITIVE))


# ---------------------------------------------------------------------------
# Per-subtype construction. Each returns:
#   structured_groups, background_slots_used, total_span_days, extra_meta
# ---------------------------------------------------------------------------

def _build_positive(rng, pattern):
    n_occ = _pick_n_occ_positive(rng)
    starts, total_span, min_gap, burst_span = _schedule_sites(rng, n_occ)
    groups = [_site_events(rng, pattern.tuple_categories, s) for s in starts]
    n_struct = pattern.k * n_occ
    extra = {"n_occ_intended": n_occ, "min_gap_used_days": min_gap,
             "burst_span_days_used": burst_span, "site_order": "canonical",
             "site_categories": [list(pattern.tuple_categories)] * n_occ}
    return groups, n_struct, total_span, extra


def _build_timing_matched(rng, pattern):
    n_occ = _pick_n_occ_positive(rng)
    starts, total_span, min_gap, burst_span = _schedule_sites(rng, n_occ)
    site_cats = [_fresh_random_tuple(rng, pattern.k, exclude_tuple=pattern.tuple_categories) for _ in starts]
    groups = [_site_events(rng, cats, s) for cats, s in zip(site_cats, starts)]
    n_struct = pattern.k * n_occ
    extra = {"n_occ_intended": n_occ, "min_gap_used_days": min_gap,
             "burst_span_days_used": burst_span, "site_order": "fresh_random_per_site",
             "site_categories": [list(c) for c in site_cats]}
    return groups, n_struct, total_span, extra


def _build_category_identity(rng, pattern):
    n_occ = _pick_n_occ_positive(rng)
    _, total_span, min_gap, burst_span = _schedule_sites(rng, n_occ)
    # same multiset as a genuine reference pattern: each of the k
    # categories appears exactly n_occ times, scattered independently in
    # time with no burst clustering and no enforced order.
    flat = list(pattern.tuple_categories) * n_occ
    rng.shuffle(flat)
    ts = np.sort(rng.uniform(0, total_span, size=len(flat)))
    groups = [[(c, float(t))] for c, t in zip(flat, ts)]  # each event its own "group" (ungrouped)
    n_struct = len(flat)
    extra = {"n_occ_intended": n_occ, "min_gap_used_days": min_gap,
             "burst_span_days_used": burst_span, "site_order": "scattered_no_structure",
             "site_categories": None}
    return groups, n_struct, total_span, extra


def _build_order_permutation(rng, pattern):
    n_occ = _pick_n_occ_positive(rng)
    starts, total_span, min_gap, burst_span = _schedule_sites(rng, n_occ)
    site_cats = [_non_canonical_permutation(rng, pattern.tuple_categories) for _ in starts]
    groups = [_site_events(rng, cats, s) for cats, s in zip(site_cats, starts)]
    n_struct = pattern.k * n_occ
    extra = {"n_occ_intended": n_occ, "min_gap_used_days": min_gap,
             "burst_span_days_used": burst_span, "site_order": "non_canonical_permutation",
             "site_categories": [list(c) for c in site_cats]}
    return groups, n_struct, total_span, extra


def _build_boundary_single_occurrence(rng, pattern):
    starts, total_span, min_gap, burst_span = _schedule_sites(rng, 1)
    groups = [_site_events(rng, pattern.tuple_categories, starts[0])]
    extra = {"n_occ_intended": 1, "min_gap_used_days": min_gap,
             "burst_span_days_used": burst_span, "site_order": "canonical",
             "site_categories": [list(pattern.tuple_categories)]}
    return groups, pattern.k, total_span, extra


def _build_boundary_tight_burst(rng, pattern):
    min_gap, burst_span = _min_gap_days()
    violating_gap = rng.uniform(0.05 * min_gap, 0.5 * min_gap)
    starts, total_span, _, _ = _schedule_sites(rng, 2, gap_override_days=violating_gap)
    groups = [_site_events(rng, pattern.tuple_categories, s) for s in starts]
    n_struct = pattern.k * 2
    extra = {"n_occ_intended": 2, "min_gap_used_days": min_gap,
             "min_gap_actual_days": float(starts[1] - starts[0]),
             "burst_span_days_used": burst_span, "site_order": "canonical_but_gap_violates_persistence",
             "site_categories": [list(pattern.tuple_categories)] * 2}
    return groups, n_struct, total_span, extra


def _build_pure_background(rng, pattern):
    # sample a span exactly like a positive would (same n_occ distribution)
    # so duration/density is not itself a giveaway for this easy subtype,
    # then discard the site placements and use only the total span.
    n_occ = _pick_n_occ_positive(rng)
    _, total_span, min_gap, burst_span = _schedule_sites(rng, n_occ)
    extra = {"n_occ_intended": 0, "min_gap_used_days": min_gap,
              "burst_span_days_used": burst_span, "site_order": "none",
              "site_categories": None}
    return [], 0, total_span, extra


_BUILDERS = {
    "positive": _build_positive,
    "timing_matched": _build_timing_matched,
    "category_identity": _build_category_identity,
    "order_permutation": _build_order_permutation,
    "boundary_single_occurrence": _build_boundary_single_occurrence,
    "boundary_tight_burst": _build_boundary_tight_burst,
    "pure_background": _build_pure_background,
}

_LABEL_FOR_SUBTYPE = {
    "positive": 1,
    "timing_matched": 0,
    "category_identity": 0,
    "order_permutation": 0,
    "boundary_single_occurrence": 0,
    "boundary_tight_burst": 0,
    "pure_background": 0,
}


def build_sample(rng, subtype: str, pattern, sample_id: str, user_id: str, split: str, seed_used: int) -> Dict[str, Any]:
    """
    Build one complete SEQUENCE_LENGTH-event sample of the given subtype.
    `pattern` is the reference PatternDef (from THIS split's pattern pool)
    - even pure_background uses one, purely to bias its background draws.
    """
    if subtype not in _BUILDERS:
        raise ValueError(f"Unknown subtype: {subtype}")

    groups, n_struct, total_span, extra = _BUILDERS[subtype](rng, pattern)
    n_bg = SEQUENCE_LENGTH - n_struct
    if n_bg < 0:
        raise RuntimeError(
            f"subtype={subtype} pattern.k={pattern.k} produced {n_struct} "
            f"structural events, exceeding SEQUENCE_LENGTH={SEQUENCE_LENGTH}."
        )
    background = _sample_background(rng, n_bg, total_span, list(pattern.tuple_categories))
    events, group_positions = _assemble(groups, background)
    assert len(events) == SEQUENCE_LENGTH, (subtype, len(events))

    label = _LABEL_FOR_SUBTYPE[subtype]

    training_example = {
        "sample_id": sample_id,
        "user_id": user_id,
        "events": events,
        "label": label,
    }

    eval_metadata = {
        "sample_id": sample_id,
        "split": split,
        "label": label,
        "subtype": subtype,
        "reference_pattern_id": pattern.pattern_id,
        "reference_pattern_categories": list(pattern.tuple_categories),
        "reference_pattern_k": pattern.k,
        "generation_seed": seed_used,
        "total_span_days": round(total_span, 4),
        "n_events": len(events),
        "group_global_positions": group_positions,  # e.g. one list per site
        **extra,
    }
    return {"training_example": training_example, "eval_metadata": eval_metadata}
