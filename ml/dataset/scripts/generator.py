"""
Dataset V2 - pattern pool construction.

Reuses the same 15-category vocabulary and the same tuple-identity-holdout
mechanism used in V1 (the V1 audit did not find that mechanism itself to be
broken: exact-tuple overlap across splits was 0/0/0). Adds one fix: category
-SET disjointness across splits (see config.py for why).
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

from config import (
    CATEGORIES, K_CHOICES, K_WEIGHTS, N_PATTERN_POOL,
    PATTERN_SPLIT_SHARE, ENFORCE_CATEGORY_SET_DISJOINT_SPLITS, BASE_SEED,
)


@dataclass(frozen=True)
class PatternDef:
    pattern_id: str
    tuple_categories: Tuple[str, ...]
    split: str

    @property
    def k(self) -> int:
        return len(self.tuple_categories)


def _sample_tuple(rng, k):
    cats = rng.choice(CATEGORIES, size=k, replace=False)
    return tuple(cats.tolist())


def build_pattern_pool(n_patterns=N_PATTERN_POOL, seed=BASE_SEED) -> List[PatternDef]:
    rng = np.random.default_rng(seed)
    seen_tuples, tuples = set(), []
    attempts = 0
    while len(tuples) < n_patterns and attempts < n_patterns * 100:
        attempts += 1
        k = int(rng.choice(K_CHOICES, p=K_WEIGHTS))
        t = _sample_tuple(rng, k)
        if t in seen_tuples:
            continue
        seen_tuples.add(t)
        tuples.append(t)
    if len(tuples) != n_patterns:
        raise RuntimeError(f"Could only generate {len(tuples)}/{n_patterns} unique pattern tuples.")

    rng.shuffle(tuples)
    n_train = int(round(n_patterns * PATTERN_SPLIT_SHARE["train"]))
    n_val = int(round(n_patterns * PATTERN_SPLIT_SHARE["val"]))
    splits = ["train"] * n_train + ["val"] * n_val
    splits += ["test"] * (n_patterns - len(splits))

    tuples = list(tuples)
    if ENFORCE_CATEGORY_SET_DISJOINT_SPLITS:
        tuples, splits = _enforce_category_set_disjointness(rng, tuples, splits)

    patterns = [
        PatternDef(pattern_id=f"pat_{i:04d}", tuple_categories=t, split=s)
        for i, (t, s) in enumerate(zip(tuples, splits))
    ]
    return patterns


def _enforce_category_set_disjointness(rng, tuples, splits, max_rounds=500):
    """
    Forbids ANY pattern's category SET (in any split) from equalling,
    being a subset of, or being a superset of a pattern's category set in
    a DIFFERENT split - checked pairwise across train/val/test, not just
    relative to train. Resamples the offending non-train tuple first
    (preferring to keep train fixed); if both sides of an offending pair
    are non-train, resamples the val-side.
    """
    priority = {"train": 0, "val": 1, "test": 2}  # lower = kept fixed longer

    def offending_pairs():
        idx_by_split = {"train": [], "val": [], "test": []}
        for i, s in enumerate(splits):
            idx_by_split[s].append(i)
        pairs = []
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            for i in idx_by_split[a]:
                si = frozenset(tuples[i])
                for j in idx_by_split[b]:
                    sj = frozenset(tuples[j])
                    if si == sj or si <= sj or sj <= si:
                        pairs.append((i, j))
        return pairs

    for _ in range(max_rounds):
        pairs = offending_pairs()
        if not pairs:
            return tuples, splits
        existing = set(tuples)
        to_resample = set()
        for i, j in pairs:
            # resample whichever side is lower priority (val before test,
            # never train unless both sides are train, which can't happen
            # here since pairs are cross-split)
            victim = i if priority[splits[i]] >= priority[splits[j]] else j
            to_resample.add(victim)
        for i in to_resample:
            k = len(tuples[i])
            for _try in range(500):
                cand = _sample_tuple(rng, k)
                if cand in existing:
                    continue
                cset = frozenset(cand)
                clashes = False
                for jdx, s2 in enumerate(splits):
                    if jdx == i or s2 == splits[i]:
                        continue
                    u = frozenset(tuples[jdx])
                    if cset == u or cset <= u or u <= cset:
                        clashes = True
                        break
                if clashes:
                    continue
                existing.discard(tuples[i])
                tuples[i] = cand
                existing.add(cand)
                break
    raise RuntimeError(
        "Could not achieve category-set-disjoint train/val/test pattern "
        "pools within the resample budget; increase N_PATTERN_POOL or the "
        "vocabulary size, or relax ENFORCE_CATEGORY_SET_DISJOINT_SPLITS."
    )


def pools_by_split(patterns: List[PatternDef]) -> Dict[str, List[PatternDef]]:
    out = {"train": [], "val": [], "test": []}
    for p in patterns:
        out[p.split].append(p)
    return out


def verify_pool_disjointness(patterns: List[PatternDef]) -> Dict:
    by = pools_by_split(patterns)
    tuple_sets = {s: set(p.tuple_categories for p in ps) for s, ps in by.items()}
    catset_sets = {s: set(frozenset(p.tuple_categories) for p in ps) for s, ps in by.items()}
    report = {"sizes": {s: len(v) for s, v in tuple_sets.items()},
              "tuple_overlaps": {}, "categoryset_relations": {}}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        report["tuple_overlaps"][f"{a}_{b}"] = len(tuple_sets[a] & tuple_sets[b])
        rel = 0
        for u in catset_sets[a]:
            for v in catset_sets[b]:
                if u == v or u <= v or v <= u:
                    rel += 1
        report["categoryset_relations"][f"{a}_{b}"] = rel
    report["all_clear"] = (
        all(v == 0 for v in report["tuple_overlaps"].values())
        and all(v == 0 for v in report["categoryset_relations"].values())
    )
    return report
