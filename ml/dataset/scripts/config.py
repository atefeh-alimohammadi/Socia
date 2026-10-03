"""
Dataset V2 configuration - Temporal Behavioral Pattern Recurrence Detection.

Every constant here defines the TASK, not a model. Nothing in this file is
tuned to raise any baseline's or V3-A's F1. See README.md for the audit
findings each constant is a direct response to.
"""

BASE_SEED = 20260925  # arbitrary, fixed, documented; changing it changes
                       # the whole dataset deterministically and reproducibly

# ---------------------------------------------------------------------------
# Behavioral event vocabulary
#
# REUSED, UNMODIFIED, from Socia's existing 15-category taxonomy as it is
# implemented in the current (V1) generator.py. The V1 audit did not find
# the vocabulary itself to be a problem, so V2 does not invent categories.
# ---------------------------------------------------------------------------
VALID_EMOTIONS = [
    "anxiety", "confidence", "fear", "excitement", "frustration",
    "sadness", "shame", "hope",
]
VALID_TAGS = [
    "avoidance", "overthinking", "self_criticism", "withdrawal",
    "reassurance_seeking", "approach_coping", "emotional_dysregulation",
]
CATEGORIES = VALID_EMOTIONS + VALID_TAGS
assert len(CATEGORIES) == 15
CATEGORY_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}

# ---------------------------------------------------------------------------
# Sequence shape
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 25  # kept equal to V1's WINDOW_SIZE, for comparability

# ---------------------------------------------------------------------------
# Pattern pool (ordered category-tuple identities)
# ---------------------------------------------------------------------------
N_PATTERN_POOL = 90
K_CHOICES = [2, 3, 4]
K_WEIGHTS = [0.25, 0.5, 0.25]
PATTERN_SPLIT_SHARE = {"train": 0.70, "val": 0.15, "test": 0.15}

# NEW in V2. V1 audit finding: pattern-identity holdout used exact-tuple
# disjointness only. Measured category-SET overlap across splits: 1
# same-set reordering + 6 subset/superset relations for val vs train, 0/2
# for test vs train. That lets a val/test pattern be solved by recognizing
# a category SET already seen in training, even though the exact ordered
# tuple is novel. V2 additionally forbids this.
ENFORCE_CATEGORY_SET_DISJOINT_SPLITS = True

# ---------------------------------------------------------------------------
# Occurrence-site construction
# ---------------------------------------------------------------------------
WITHIN_OCC_GAP_HOURS_RANGE = (0.05, 6.0)

# V1 audit measured actual occurrence spans up to 16.25h even though V1's
# generator nominally allowed up to 72h (W_PATTERN_HOURS). V2 sets an
# honest, enforced cap instead of an unused nominal one.
BURST_SPAN_HOURS_CAP = 24.0

# ---------------------------------------------------------------------------
# RECURRENCE DEFINITION - the central methodological change from V1
#
# A sequence is POSITIVE iff it contains >= MIN_OCCURRENCES_FOR_RECURRENCE
# temporally-separated "sites", each an instance of the SAME ordered
# category tuple, where:
#
#   (a) each site independently satisfies the occurrence-level definition:
#       the tuple's categories appear in canonical order, all k events
#       within BURST_SPAN_HOURS_CAP of each other ("what is one instance").
#
#   (b) every consecutive pair of sites is separated by a gap that is BOTH
#         >= MIN_INTER_OCC_GAP_DAYS_ABS   (an absolute floor), AND
#         >= PERSISTENCE_RATIO * burst_span_days_used_by_that_pair
#       i.e. recurrence must be spread out relative to the internal scale
#       of a single instance - not one elongated burst that happens to
#       repeat a tuple. This operationalizes "temporal persistence"
#       (candidate Definition B) using a ratio to the sample's own scale
#       rather than an arbitrary absolute rule such as "3 occurrences
#       within 14 days".
#
# A sequence containing exactly one well-formed site is explicitly NOT
# positive. This directly fixes the single largest V1 audit finding: 54.6%
# of V1's "positive" windows contained exactly one true occurrence, i.e.
# V1's label was mostly testing "does this window contain ONE instance of
# a pattern", not recurrence.
# ---------------------------------------------------------------------------
MIN_OCCURRENCES_FOR_RECURRENCE = 2
N_OCC_CHOICES_POSITIVE = [2, 3, 4]
N_OCC_WEIGHTS_POSITIVE = [0.5, 0.35, 0.15]

MIN_INTER_OCC_GAP_DAYS_ABS = 2.0
PERSISTENCE_RATIO = 3.0
INTER_OCC_JITTER_LOGNORMAL_SIGMA = 0.25

# ---------------------------------------------------------------------------
# Background noise
#
# The SAME background-generation function and the SAME PATTERN_VOCAB_BIAS
# policy is applied to every subtype, positive or negative, tied to a
# "reference_pattern_id" that every sample carries (even pure-background
# negatives get one, used only to bias background category draws). This is
# what stops "background composition" itself from correlating with the
# label, which the V1 audit found to be a major confound (V1's negatives
# were dominated by a separate 40%-of-users "control" population with a
# systematically different background category mix and P(label=1 |
# control user) == 0 by construction).
# ---------------------------------------------------------------------------
PATTERN_VOCAB_BIAS_PROB = 0.3

# ---------------------------------------------------------------------------
# Dataset composition - target proportions
#
# NOT optimized for F1. Chosen so that (a) no single negative subtype
# dominates the way V1's background/control negatives did (~70-80% of V1
# negatives were near-pure background), and (b) every subtype has enough
# examples for separate evaluation (Q1-Q6 in the audit brief).
# ---------------------------------------------------------------------------
POSITIVE_SHARE = 0.50

NEGATIVE_SUBTYPE_SHARES = {
    # Same site count / timing / gaps / duration / density as a matched
    # positive; each site gets an independent fresh random tuple, so no
    # identity recurs anywhere. Destroys recurrence, preserves timing exactly.
    "timing_matched": 0.22,

    # Same category MULTISET as a genuine reference pattern (k categories,
    # each repeated n_occ times - literally the same "ingredients" a
    # positive would use) but scattered at independent random times with no
    # burst clustering and no enforced order. Tests bag-of-category shortcuts.
    "category_identity": 0.16,

    # Same site timing/gaps/span and same per-site category SET as a
    # genuine positive, but each site's within-burst order is a permutation
    # of the canonical tuple order. Directly reusable with the existing
    # permutation/inconsistency diagnostics.
    "order_permutation": 0.20,

    # Exactly ONE genuine, well-formed occurrence. The central boundary
    # case: satisfies the occurrence-level definition perfectly, fails
    # recurrence only because count < MIN_OCCURRENCES_FOR_RECURRENCE.
    "boundary_single_occurrence": 0.20,

    # >=2 genuine, well-formed occurrences, but the inter-occurrence gap is
    # forced below the persistence threshold - a burst that repeats a
    # tuple twice in quick succession, not temporally persistent recurrence.
    "boundary_tight_burst": 0.12,

    # No injected structure at all. Kept as a small, explicit minority
    # (not the ~70-80% majority it was in V1) so a majority/background
    # baseline remains checkable without dominating the benchmark.
    "pure_background": 0.10,
}
assert abs(sum(NEGATIVE_SUBTYPE_SHARES.values()) - 1.0) < 1e-9

TARGET_TOTAL_PER_SPLIT = {"train": 12000, "val": 2000, "test": 2000}

# ---------------------------------------------------------------------------
# Users
#
# Every sample belongs to exactly one synthetic "user session", and every
# user is assigned to exactly one split at creation time (disjoint by
# construction, via a split-namespaced id, not by shuffling shared windows
# post hoc the way V1 sliced many overlapping windows from one long
# multi-pattern per-user timeline).
# ---------------------------------------------------------------------------
SAMPLES_PER_USER_RANGE = (1, 3)

# ---------------------------------------------------------------------------
# Counterfactual evaluation set (test split only)
# ---------------------------------------------------------------------------
N_COUNTERFACTUAL_BASE_SAMPLES = 300
COUNTERFACTUAL_TRANSFORMS = [
    "order_permutation",       # permute within-site order -> expect flip to negative
    "timestamp_collapse",      # collapse inter-site gaps below persistence floor
    "identity_substitution",   # replace one site's tuple with a fresh random one
]
