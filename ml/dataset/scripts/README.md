# Dataset V2 — Temporal Behavioral Pattern Recurrence Detection

This is a from-scratch reimplementation, not a rename of V1. V1 is untouched
under `research/dataset/v1/` (whatever your existing path is — this
pipeline never reads or writes there).

## 1. Recurrence definition actually implemented

A sequence is **positive** iff it contains `>= MIN_OCCURRENCES_FOR_RECURRENCE`
(= 2) temporally-separated **sites**, each an instance of the **same**
ordered category tuple, where:

- **(a) each site** independently satisfies the occurrence-level definition:
  the tuple's categories appear in canonical order, all *k* events within
  `BURST_SPAN_HOURS_CAP` (24h) of each other — "what one instance is."
- **(b) every consecutive pair of sites** is separated by a gap that is
  both `>= MIN_INTER_OCC_GAP_DAYS_ABS` (2 days, an absolute floor) **and**
  `>= PERSISTENCE_RATIO × burst_span_days` (3×) — recurrence must be spread
  out relative to the internal scale of one instance, not one elongated
  burst that happens to repeat a tuple.

This is Definition B ("repeated event + temporal persistence") made
concrete with a *scale-relative* rule instead of an arbitrary absolute one
("3 occurrences within 14 days"). A sequence with exactly one well-formed
site is **not** positive — this directly fixes the single largest V1 audit
finding: 54.6% of V1's "positive" windows contained exactly one true
occurrence, meaning V1 mostly tested "does this window contain one
instance", not recurrence.

## 2. Sample construction — why it's not V1 + a rename

V1 simulated one long (120-day) per-user event timeline and then sliced it
with a stride-1 sliding window. The audit found this caused:
- ~97.7% of selected windows overlapping another selected window of the
  same user (severe pseudo-replication),
- a separate 40%-of-users "control" population supplying 66–80% of V1's
  negatives, with `P(label=1 | control user) == 0` by construction — i.e.
  "is this user a control" was itself a usable shortcut,
- background category composition differing systematically between
  control and pattern users (`cat_entropy` alone: test AUC 0.819 in V1,
  driven mostly by this confound — it drops to 0.65 once restricted to
  pattern-users-only).

V2 builds **each 25-event sequence directly**, with no long shared
timeline and no separate control-user population. Every sample — positive
or negative, of any subtype — is generated through the exact same
background-noise function and the same `PATTERN_VOCAB_BIAS_PROB` policy,
tied to a `reference_pattern_id` every sample carries (even
`pure_background` negatives get one, used only to bias background draws).
This removes the population-level confound at the root instead of
patching it with post-hoc balancing.

## 3. Positive and negative construction

| Subtype | Construction | What it targets |
|---|---|---|
| `positive` | 2–4 genuine sites of the **same** tuple, persistence-satisfying gaps | the task itself |
| `timing_matched` | Same site count/timing/gaps/span as a matched positive; **each site gets an independent fresh random tuple** | destroys recurring identity, keeps timing/count/density/gap distribution exactly matched |
| `category_identity` | Same category **multiset** as a genuine reference pattern (each of *k* categories repeated `n_occ` times), scattered at independent random times, **no burst clustering, no enforced order** | bag-of-category shortcut ("right ingredients, no structure") |
| `order_permutation` | Same site timing/gaps/span and same per-site category **set** as a genuine positive; each site's **within-burst order is a non-canonical permutation** | order-sensitivity; directly reusable with your permutation/inconsistency diagnostics |
| `boundary_single_occurrence` | Exactly **one** genuine, well-formed occurrence | the central boundary case — passes occurrence-level structure perfectly, fails only on count |
| `boundary_tight_burst` | **Two** genuine occurrences, but the inter-occurrence gap is forced **below** the persistence threshold | "burst vs. temporally-persistent recurrence" (your section 10D) |
| `pure_background` | No injected structure; span sampled the same way as a positive's would-be span, then discarded, so duration isn't itself a giveaway | small explicit minority (10% of negatives / 5% of total), **not** the ~70–80% majority it was in V1 |

No arbitrary `count(event_type) >= N` rule anywhere. Every subtype is
built by the same machinery a positive is built by, varying exactly one
axis (identity, order, count, or persistence) at a time.

## 4. Measured verification (this was actually run, not just designed)

Generated at target scale (train 12,000 / val 2,000 / test 2,000) and
checked with `diagnostics.py`. Regenerating from scratch reproduces the
train/test/pattern-pool/counterfactual files **byte-for-byte** (verified).

- **Pattern pool**: 63/14/13 train/val/test patterns, exact-tuple overlap
  0/0/0 **and** category-set subset/superset/equality relations 0/0/0
  across all three split pairs (V1 had 1 same-set + 6 subset/superset
  relations for val vs. train — that's now closed).
- **User overlap across splits**: 0/0/0 (structurally guaranteed, not
  just measured after the fact).
- **Class balance**: exactly 50/50 in every split. **Max negative-subtype
  share**: 22% (`timing_matched`) — no subtype dominates (V1's dominant
  subtype was ~74–80% of negatives).
- **Shortcut baselines on test** (same methodology as the V1 audit — train
  on train, threshold on val, score on test):

  | Baseline | V1 ROC-AUC | V2 ROC-AUC |
  |---|---|---|
  | timing-only | 0.818 | **0.734** |
  | category-only | 0.751 | **0.674** |
  | timing+category | 0.883 | **0.809** |

  Every shortcut baseline dropped substantially. `cat_entropy` alone:
  0.819 → **0.681**. `duration_h` alone: 0.688 → **0.656**.
- **Per-subtype false-positive rate of the timing+category baseline on
  test** — this is the important diagnostic, because it shows the
  baseline's remaining accuracy is *not* evenly earned:

  | Negative subtype | FP rate |
  |---|---|
  | `boundary_single_occurrence` | 0.08 |
  | `boundary_tight_burst` | 0.21 |
  | `pure_background` | 0.49 |
  | `timing_matched` | 0.37 |
  | `category_identity` | 0.71 |
  | `order_permutation` | **0.93** |

  A timing+category (bag-of-features) model gets `order_permutation` and
  `category_identity` wrong the large majority of the time — by
  construction, since those subtypes are *identical* to a matched
  positive on every timing and category-count feature and differ only in
  order/temporal arrangement. Any model that wants a strong score on V2
  has to actually use sequence order and temporal placement, not just
  aggregate statistics. This is the direct, measured answer to your Q2–Q4.

Full machine-readable numbers: `data/diagnostics_report.json` after
running the generator (see below).

## 5. Directory structure

```
research/dataset/v2/
  config.py            # every task constant, with audit-finding provenance
  generator.py         # pattern pool (tuple + category-set disjoint splits)
  simulate_user.py      # per-sample construction, all 7 subtypes
  counterfactuals.py    # order/timestamp/identity counterfactual transforms
  build_dataset.py      # main pipeline; writes everything under data/
  diagnostics.py        # standalone, read-only verification (run any time)
  README.md             # this file
  data/                 # generated output (gitignored / regenerable)
    train.jsonl
    val.jsonl
    test.jsonl
    train_eval_metadata.jsonl
    val_eval_metadata.jsonl
    test_eval_metadata.jsonl
    pattern_pool.json
    counterfactual_eval.jsonl
    generation_report.json
    diagnostics_report.json   # written by diagnostics.py
```

## 6. Commands

```bash
cd research/dataset/v2
python build_dataset.py          # generates everything under data/
python diagnostics.py data       # runs all verification checks, prints +
                                  # writes data/diagnostics_report.json
```

Both are deterministic given `config.BASE_SEED` (default `20260925`).
Changing `BASE_SEED` regenerates a different but internally consistent
dataset; the same seed always reproduces the same files byte-for-byte
(empirically verified above; no `random.seed()`-free calls, no
process-randomized `hash()` anywhere — see the `_stable_seed` helper in
`counterfactuals.py`, which uses `zlib.crc32` specifically because
Python's built-in `hash()` on strings is *not* reproducible across
processes unless `PYTHONHASHSEED` is fixed, a hazard your own
`check_reproducibility.py` script is built to catch).

## 7. Output schema

**`{split}.jsonl`** — model input only:
```json
{"sample_id": "train_000123", "user_id": "train_user_00042",
 "events": [{"category_id": 1, "category": "confidence", "delta_hours": 0.0}, ...],
 "label": 1}
```
Exactly `SEQUENCE_LENGTH` (25) events. **Nothing else is in this file** —
no subtype, no pattern id, no seed. This is what a model may see.

**`{split}_eval_metadata.jsonl`** — verification/analysis only, joined by
`sample_id`, **never fed to a model**:
```json
{"sample_id": "train_000123", "split": "train", "label": 1,
 "subtype": "positive", "reference_pattern_id": "pat_0031",
 "reference_pattern_categories": ["fear", "hope"], "reference_pattern_k": 2,
 "generation_seed": 20260931123, "total_span_days": 11.37, "n_events": 25,
 "group_global_positions": [[3, 7], [14, 19]],
 "n_occ_intended": 2, "min_gap_used_days": 3.0, "burst_span_days_used": 1.0,
 "site_order": "canonical", "site_categories": [["fear","hope"], ["fear","hope"]]}
```

**`pattern_pool.json`**: `{pattern_id, categories, k, split}` per pattern.

**`counterfactual_eval.jsonl`** (test only): one `role: "base"` row per
selected positive plus one `role: "transformed"` row per applied
transform, linked by `counterfactual_group_id` and `base_sample_id`.
`expected_label` on transformed rows states what the transform is
*designed* to do, not a measured model outcome.

**`generation_report.json`**: pattern-pool disjointness report, exact
subtype plan per split, user-overlap check, counterfactual row count —
the config-time self-check, analogous to V1's `generation_report.json`
but with the checks V1 was missing.

## 8. Built-in validation (`diagnostics.py`)

Re-derives, read-only, from the written files: class/subtype balance per
split and the max negative-subtype share; user-disjointness; pattern-pool
tuple **and** category-set disjointness; duration/gap/event-count/category
distributions by label and by subtype; the recurrence `n_occ_intended`
distribution among positives; single-feature AUC screen; the three
shortcut baselines (timing-only / category-only / combined) trained on
train, thresholded on val, scored on test; and the combined baseline's
per-negative-subtype false-positive rate. This is exactly the audit
methodology used on V1, so V1 and V2 numbers are directly comparable (see
section 4).

## 9. How each V1 audit finding is addressed

| V1 audit finding | V2 fix |
|---|---|
| 54.6% of "positives" had exactly 1 true occurrence — V1 tested single-occurrence presence, not recurrence | Recurrence now requires `>= 2` sites by definition; single-occurrence sequences are the explicit `boundary_single_occurrence` **negative** subtype |
| True occurrence span capped at 16.25h even though the nominal cap was 72h (unused headroom) | `BURST_SPAN_HOURS_CAP = 24h`, honest and enforced |
| No explicit "burst vs. persistent recurrence" boundary case | `boundary_tight_burst`: 2 genuine occurrences with the persistence gap deliberately violated |
| Control-user population supplied 66–80% of negatives; `P(label=1\|control)=0` was itself a shortcut | No control-user population; every sample (of any subtype) is built through the identical background/bias machinery |
| `cat_entropy` alone: test AUC 0.819, largely a control-vs-pattern-user artifact | 0.681 in V2 (measured), and the artifact it rode on no longer exists |
| ~97.7% of selected windows overlapped another selected window of the same user (pseudo-replication) | Samples are built directly, not sliced from a shared timeline; no sliding window at all |
| Hard negatives (`reordered`, `too_slow`) were ≤11% of negatives; easy background dominated | 6 explicit negative subtypes, each capped, none exceeding 22% of negatives; `pure_background` capped at 10% |
| Category-SET (not just exact-tuple) overlap across pattern splits: 1 same-set + 6 subset/superset relations for val vs. train | `ENFORCE_CATEGORY_SET_DISJOINT_SPLITS` closes this to 0/0/0, verified |
| No counterfactual/intervention evaluation set | `counterfactual_eval.jsonl`: order-permutation, timestamp-collapse, identity-substitution transforms, each linked to its base sample |
| Reproducibility only "not disproven" by static inspection | Regeneration verified **byte-for-byte identical**; no unseeded RNG source (`hash()` avoided in favor of `zlib.crc32`) |

## 10. What this does *not* do (by design, per your instructions)

- Does not touch V1's code or data.
- Does not retrain or evaluate V3-A.
- Is not tuned to maximize any baseline's or V3-A's F1 — subtype
  proportions and thresholds above were chosen for coverage and balance,
  not to hit a target score. The measured drop in shortcut-baseline AUC
  (0.883 → 0.809 combined) is a *consequence* of removing confounds, not
  a target that was optimized for.
- Does not implement Stage 2 (behavioral significance/meaningfulness) —
  V2 is Stage 1 (recurrence detection) only, as scoped.
