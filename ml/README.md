# Socia Research — Temporal Behavioral Pattern Recurrence Detection

An independent ML research module investigating whether a model can distinguish **operationally defined temporal recurrence** in a longitudinal sequence of behavioral events from sequences that are superficially similar in timing, category composition, or event order.

This is a **research module**, separate from Socia's production system. See [Research vs. Production](docs/problem_definition.md#research-vs-production-boundary).

The research does not make clinical, diagnostic, or general claims about human behavior. Instead, it evaluates recurrence-sensitive models on a controlled synthetic benchmark designed to isolate specific sequence-level properties.

The research system is currently **offline and experimental**. It is **not integrated yet** into Socia's production pattern-detection pipeline.

---

## What is this?

Socia's production system currently extracts structured behavioral observations from conversation and promotes an observation group to a user-visible pattern using a deterministic application-level rule.

The research module investigates a narrower question:

> **Can a model distinguish operationally defined temporal recurrence from superficially similar sequences that differ in timing, category composition, or event structure?**

The research models are **not currently used by the production application**. They do not influence the production 3+-observation promotion rule and do not generate user-facing patterns.

The benchmark defines recurrence synthetically and operationally. Therefore, performance on this benchmark should not be interpreted as validated detection of real-world psychological or behavioral traits.

---

## Research question

> Given a 25-event window of behavioral events, can a model distinguish a genuinely recurring ordered category pattern — the same category tuple appearing at least twice as temporally separated sites — from sequences that are superficially similar but do not satisfy the benchmark's recurrence definition?

The first iteration, **V1**, exposed several dataset shortcuts. The benchmark was subsequently redesigned as **V2** to address those issues and provide a more controlled evaluation setting.

The research now covers three broad modeling approaches:

1. **Aggregate feature baseline** — timing and category features without explicit sequence modeling.
2. **Supervised temporal sequence models** — V2 and V3 Transformer variants trained specifically for the benchmark.
3. **Zero-shot LLM reference** — Qwen2.5-7B evaluated on the same frozen V2 test set with a fixed prompt and no task-specific fine-tuning.

Additional experiments examine recurrence representations, controlled counterfactuals, recurrence count, and error structure.

---

# Key results

The current V2 benchmark uses the same frozen test set for the timing + category baseline and all V2-family neural models:

* **2,000 test examples**
* **1,000 positive**
* **1,000 negative**

Thresholds for the reported operating-point metrics were selected using validation data only.

V1 is **not included in this comparison** because it was evaluated on the earlier V1 benchmark.

| Model                          |   ROC-AUC |    PR-AUC |        F1 | Precision |    Recall |  Accuracy |
| ------------------------------ | --------: | --------: | --------: | --------: | --------: | --------: |
| **Timing + category baseline** | **0.807** | **0.759** | **0.762** |     0.662 | **0.899** | **0.720** |
| V2                             |     0.747 |     0.701 |     0.729 |     0.635 |     0.856 |     0.682 |
| V3-A                           |     0.774 |     0.722 |     0.731 | **0.688** |     0.779 | **0.713** |
| V3-C                           |     0.770 |     0.716 | **0.736** |     0.659 |     0.834 |     0.702 |
| V3-A Bucketed                  |     0.776 | **0.730** |     0.720 |     0.677 |     0.769 |     0.701 |

A separate Qwen2.5-7B zero-shot experiment is reported below because it is a different modeling setup rather than another supervised sequence-model baseline.

### Main result

The timing + category baseline exceeds all current neural models on **ROC-AUC and PR-AUC** on the V2 test set.

At the same time, controlled interventions show that the neural models depend substantially on the recurrence-related representations supplied to them. Removing or disrupting those representations causes large drops in ranking performance.

These findings should be considered together:

> **The current experiments do not show that neural sequence modeling outperforms simple aggregate timing/category features. They do show that the trained neural models make substantial use of their recurrence-related representations.**

The latter is evidence of **model dependence on the supplied representation**, not proof that the models have learned a validated or psychologically meaningful concept of behavioral recurrence.

For detailed results and diagnostics, see [`docs/evaluation.md`](docs/evaluation.md).

---

# Research pipeline

```mermaid
flowchart LR
    A["V2 dataset generator<br/>(controlled positives +<br/>6 negative subtypes)"] --> B["Dataset audit<br/>(balance, leakage,<br/>reproducibility)"]

    B --> C["Timing + category<br/>baseline"]
    B --> D["V2 / V3-A / V3-C /<br/>V3-A Bucketed"]

    C --> E["Final evaluation<br/>(frozen test set,<br/>validation thresholds)"]
    D --> E

    K["Qwen2.5-7B<br/>zero-shot reference"] --> E

    E --> F["Subtype analysis"]
    E --> G["Recurrence-representation<br/>ablations"]
    E --> H["Counterfactual analysis"]
    E --> I["Recurrence-count analysis"]
    E --> J["Error analysis"]
```

---

# Dataset at a glance

The current primary benchmark is **V2**.

|                                   |   Train | Validation |    Test |
| --------------------------------- | ------: | ---------: | ------: |
| Total examples                    |  12,000 |      2,000 |   2,000 |
| Positive                          |   6,000 |      1,000 |   1,000 |
| Negative                          |   6,000 |      1,000 |   1,000 |
| Class balance                     | 50 / 50 |    50 / 50 | 50 / 50 |
| Max single negative subtype share |    ≤22% |       ≤22% |    ≤22% |
| User overlap across splits        |       0 |          0 |       0 |
| Pattern tuple overlap             |       0 |          0 |       0 |
| Category-set overlap              |       0 |          0 |       0 |

Six deliberately constructed negative subtypes are used:

* `timing_matched`
* `order_permutation`
* `boundary_single_occurrence`
* `category_identity`
* `boundary_tight_burst`
* `pure_background`

These subtypes were introduced to test specific shortcut hypotheses identified during V1 analysis.

See [`docs/dataset.md`](docs/dataset.md) for construction, examples, and audit details.

---

# V1 — the original benchmark

V1 is retained as historical development because its audit motivated the redesign.

The audit identified several problematic shortcut opportunities, including:

* approximately 54.6% of positive windows containing only one intended recurrence;
* substantial negative examples originating from a separate control-user population;
* strong performance from simple category/timing statistics;
* extensive overlap between selected windows;
* category-set relationships that could provide shortcut information.

V2 was redesigned in response to these findings.

V1's final test result was:

| Metric    |     V1 |
| --------- | -----: |
| ROC-AUC   | 0.6752 |
| PR-AUC    | 0.6147 |
| F1        | 0.7251 |
| Precision | 0.5825 |
| Recall    | 0.9600 |
| Accuracy  | 0.6360 |

These results are **not directly comparable with the V2-family results** because the benchmark generation and label construction changed.

See [`docs/experiment_history.md`](docs/experiment_history.md).

---

# Model progression

## V2

V2 uses:

* category embeddings;
* temporal features;
* learned relative-time attention bias;
* manual self-attention;
* learned attention pooling;
* classification head.

Its temporal representation includes:

```text
abs_norm
gap_prev_norm
gap_same_cat_norm
has_prev_same_category
```

The training protocol also includes auxiliary ranking/reordering losses.

Best validation checkpoint:

* epoch: **25**
* validation F1: **0.7380**
* threshold: **0.25**

---

## V3-A

V3-A builds on the V2 recurrence representation and adds a learned event-order embedding.

It uses plain BCE training.

Best validation checkpoint:

* epoch: **33**
* validation F1: **0.7576**
* threshold: **0.45**
* precision: **0.6873**
* recall: **0.8440**

---

## V3-C

V3-C is a controlled ablation of V3-A.

Its architectural difference is the removal of the learned event-order embedding.

Best validation checkpoint:

* epoch: **29**
* validation F1: **0.7686**
* threshold: **0.40**
* precision: **0.6696**
* recall: **0.9020**

The final test results do not show a uniform benefit from the order embedding:

* V3-A has higher ROC-AUC and PR-AUC;
* V3-C has higher F1 and recall.

The result is therefore a trade-off rather than a universal improvement.

---

## V3-A Bucketed

V3-A Bucketed replaces the continuous same-category recurrence representation with a learned discrete recurrence-gap representation.

The seven buckets are:

* `NO_PRIOR`
* `0-6h`
* `6-24h`
* `24-72h`
* `72-168h`
* `168-336h`
* `>336h`

Best validation checkpoint:

* epoch: **34**
* validation F1: **0.7716**
* threshold: **0.45**
* precision: **0.6965**
* recall: **0.8650**

---

# Timing + category baseline

The main non-neural baseline is a `HistGradientBoostingClassifier`.

It uses aggregate timing and category features, including:

* total duration;
* timing-gap statistics;
* counts under multiple gap thresholds;
* category counts;
* number of unique categories;
* maximum category count;
* Shannon entropy.

It does **not** explicitly model event order or recurrence-site identity.

On the frozen V2 test set:

* ROC-AUC: **0.807**
* PR-AUC: **0.759**
* F1: **0.762**
* precision: **0.662**
* recall: **0.899**
* accuracy: **0.720**

This baseline is an important result rather than merely a control. It shows that aggregate timing and category information remains sufficient to obtain strong performance on the current synthetic benchmark.

---

# Recurrence-representation ablations

One of the main additional questions is:

> **How much do the trained neural models depend on their explicit recurrence-related representations?**

Two interventions were used.

## Option A — no-prior recurrence representation

The continuous recurrence channels

```text
gap_same_category_norm
has_prev_same_category
```

were replaced with a no-prior representation.

ROC-AUC changed as follows:

| Model | Original | No-prior |       Δ |
| ----- | -------: | -------: | ------: |
| V2    |   0.7470 |   0.4012 | -0.3458 |
| V3-A  |   0.7738 |   0.6520 | -0.1218 |
| V3-C  |   0.7698 |   0.6526 | -0.1172 |

The large changes indicate substantial dependence on the recurrence representation.

For V2, V3-A, and V3-C, the original validation-derived threshold produced F1 = 0 after the intervention because the output distribution shifted strongly downward.

This should **not** be interpreted as complete loss of ranking information. Selecting a new threshold using the test labels can recover non-zero F1, but that is a diagnostic rather than a valid primary evaluation.

The primary interpretation therefore focuses on the degradation in ranking metrics.

---

## Option B — global recurrence-feature permutation

A second intervention globally shuffled recurrence-feature pairs across test events using a fixed seed.

For the continuous models, the shuffled pair was:

```text
(gap_same_category_norm, has_prev_same_category)
```

This preserves the marginal feature distribution while disrupting its original event-level alignment.

ROC-AUC changed as follows:

| Model | Original | Permuted |       Δ |
| ----- | -------: | -------: | ------: |
| V2    |   0.7470 |   0.4882 | -0.2588 |
| V3-A  |   0.7738 |   0.5142 | -0.2596 |
| V3-C  |   0.7698 |   0.5508 | -0.2190 |

The degradation is consistent with the models relying on the event-level alignment between recurrence features and sequence context.

However, this remains an intervention-based sensitivity result rather than proof of causal or semantic understanding.

---

## Bucketed recurrence permutation

For V3-A Bucketed, recurrence bucket IDs were globally shuffled.

This is related to Option B but is **not identical** to continuous-feature permutation.

```text
Original ROC-AUC: 0.7756
Bucket permutation ROC-AUC: 0.6278
Δ: -0.1478
```

The degradation indicates that the bucket representation also contributes predictive information in its original event-level alignment.

The experiment does not establish why its degradation differs from the continuous-feature interventions.

---

# Counterfactual analysis

Counterfactual experiments were run on **300 genuine positive test examples**.

The transformations were:

* `order_permutation`
* `timestamp_collapse`
* `identity_substitution`

These experiments measure **output sensitivity to controlled synthetic interventions**. They do not establish causal or semantic understanding.

## Order permutation

Expected-direction rates:

| Model             | Expected-direction rate |
| ----------------- | ----------------------: |
| Timing + category |                    0.0% |
| V3-A              |                   48.3% |
| V3-C              |                   53.3% |

The neural models therefore show approximately chance-level output sensitivity to this particular order-scrambling intervention.

This does not support a strong claim that the current models reliably respond to internal event-order changes.

## Timestamp collapse

| Model             | Expected-direction rate |
| ----------------- | ----------------------: |
| Timing + category |                   49.3% |
| V3-A              |                   61.0% |
| V3-C              |                   63.7% |

V3-A and V3-C show greater output sensitivity than the baseline under this intervention, although the effects remain modest.

## Identity substitution

| Model             | Expected-direction rate |
| ----------------- | ----------------------: |
| Timing + category |                   82.0% |
| V3-A              |                   82.7% |
| V3-C              |                   82.0% |

Because identity substitution also changes category composition, this intervention is not a clean test of sequence-specific reasoning.

Overall, these results show different levels of sensitivity to controlled perturbations, but they should not be interpreted as evidence of semantic understanding.

---

# Recurrence-count analysis

Positive examples were grouped by `n_occ_intended`, the number of genuine recurrence sites.

| Intended sites |   N | Baseline recall | V3-A recall | V3-C recall |
| -------------: | --: | --------------: | ----------: | ----------: |
|              2 | 487 |           0.842 |       0.694 |       0.756 |
|              3 | 360 |           0.950 |       0.828 |       0.892 |
|              4 | 153 |           0.961 |       0.935 |       0.948 |

All three systems show increasing recall as the number of recurrence sites increases.

However, the same monotonic trend appears in the aggregate baseline.

Therefore, this analysis does **not** establish that the neural models specifically learned recurrence counting.

---

# Subtype and error analysis

Historical subtype analysis showed that `order_permutation` was the largest source of false positives for all compared systems.

False-positive rates:

| Negative subtype             | Baseline |  V3-A |  V3-C |
| ---------------------------- | -------: | ----: | ----: |
| `order_permutation`          |    0.915 | 0.785 | 0.840 |
| `category_identity`          |    0.700 | 0.500 | 0.631 |
| `pure_background`            |    0.450 | 0.220 | 0.380 |
| `timing_matched`             |    0.355 | 0.355 | 0.441 |
| `boundary_tight_burst`       |    0.208 | 0.092 | 0.142 |
| `boundary_single_occurrence` |    0.085 | 0.025 | 0.050 |

These results are useful for understanding error structure, but they should not be interpreted as evidence that the neural models are globally superior to the baseline.

In particular, lower subtype FPRs can coexist with lower positive recall, as seen in the main V2 evaluation.

---

# Zero-shot Qwen2.5-7B reference experiment

A separate experiment evaluated **Qwen2.5-7B** zero-shot on the same frozen V2 test set using one fixed prompt.

Protocol:

* one model;
* one fixed prompt;
* zero-shot inference;
* no task-specific fine-tuning;
* no few-shot examples;
* no prompt optimization.

Results:

| Metric    | Qwen2.5-7B |
| --------- | ---------: |
| Accuracy  |      0.508 |
| Precision |      0.504 |
| Recall    |      0.999 |
| F1        |      0.670 |
| ROC-AUC   |      0.521 |

Confusion matrix:

```text
TN = 17
FP = 983
FN = 1
TP = 999
```

Runtime:

* total: **6,993.44 seconds**
* approximately **3.50 seconds/sample**

The model predicted the positive class for **1,982 of 2,000 examples**.

Because the benchmark is balanced, this behavior is close to a constant-positive classifier.

The ROC-AUC of **0.521** likewise indicates near-chance ranking discrimination under this specific configuration.

This result is informative as a reference point, but it should not be generalized to LLMs as a class. It does not establish that:

* LLMs cannot solve the task;
* Qwen2.5-7B lacks the underlying capability;
* another prompt would behave similarly;
* few-shot prompting would not help;
* fine-tuning would produce the same result.

---

# What the experiments currently show

The current evidence supports three main observations.

### 1. Aggregate features remain highly competitive

The timing + category baseline achieves higher ROC-AUC and PR-AUC than the neural models on V2.

The current benchmark therefore does not demonstrate that Transformer-based sequence modeling is superior to aggregate timing/category features.

### 2. Neural models depend substantially on recurrence representations

Removing or globally permuting recurrence-related features produces substantial ranking degradation across V2, V3-A, and V3-C.

This is evidence of **model dependence on the supplied recurrence representation**.

### 3. Dependence is not equivalent to validated recurrence understanding

The benchmark itself defines the recurrence structure.

Therefore, the ablations demonstrate sensitivity to representations associated with benchmark recurrence, but they do not establish that a model has learned a validated real-world behavioral concept.

The zero-shot Qwen2.5-7B result should be interpreted separately as a reference experiment for one model and one prompt.

---

# Limitations

* **Synthetic data only** — no real longitudinal user data has been used.
* **Benchmark-generator dependence** — models may exploit regularities specific to the synthetic generator.
* **No neural performance advantage over the timing + category baseline** on the primary ranking metrics.
* **No real-world validation** — generalization to real behavioral sequences is unknown.
* **No calibrated probability interpretation** — model scores are not validated probabilities of recurrence.
* **Synthetic counterfactuals** — intervention results do not establish causal or semantic understanding.
* **Order sensitivity remains unresolved** under the current counterfactual protocol.
* **Representation dependence is not psychological validity**.
* **V2 and V3 training protocols differ**: V2 uses additional auxiliary losses, while V3-A/V3-C use plain BCE.
* **Continuous and bucketed recurrence permutations are different interventions** and should not be treated as identical.
* **Threshold-dependent metrics depend on the selected operating point**.
* **Best-test-threshold F1 is diagnostic only** because it uses test labels.
* **The zero-shot LLM experiment uses one model and one prompt**.
* **The research module is not integrated into production yet**.

---

# Research vs. Production boundary

```text
SOCIA
│
├── Production
│   ├── LLM signal extraction
│   ├── deterministic pattern promotion
│   ├── episodic memory
│   ├── synthesized memories
│   └── Journeys
│
└── Research
    ├── synthetic recurrence benchmark
    ├── V1
    ├── V2
    ├── V3-A
    ├── V3-C
    ├── V3-A Bucketed
    ├── timing + category baseline
    ├── recurrence ablations
    ├── counterfactual tests
    └── zero-shot LLM reference
```

The research models currently have **no influence on production predictions, memory, Journeys, or user-facing pattern detection**.

The intended future direction is to evaluate the research models further and, if a suitable model meets the required validation criteria, connect it to the production pattern-detection pipeline.

Until then, the production system continues to use its current deterministic promotion rule.

---

# Documentation

| Document                                                                     | Contents                                                                                                      |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| [`docs/problem_definition.md`](docs/problem_definition.md)                   | Research question, operational recurrence definition, research-vs-production boundary                         |
| [`docs/dataset.md`](docs/dataset.md)                                         | V2 construction, six negative subtypes, examples, audit, and reproducibility                                  |
| [`docs/methodology.md`](docs/methodology.md)                                 | Baseline, V2/V3 architectures, training protocols, and checkpoints                                            |
| [`docs/evaluation.md`](docs/evaluation.md)                                   | Main results, ablations, counterfactuals, recurrence-count and subtype analysis, and LLM reference experiment |
| [`docs/experiment_history.md`](docs/experiment_history.md)                   | V1 → audit → V2 redesign → model progression → final evaluation                                               |
| [`docs/limitations_and_future_work.md`](docs/limitations_and_future_work.md) | Scientific limitations and future validation roadmap                                                          |

---

# Reproducibility

The current benchmark uses deterministic generation with:

```text
Dataset seed: 20260925
Recurrence permutation seed: 20260829
```

The V2 dataset was regenerated and verified byte-identical to the corresponding generation and diagnostic artifacts.

Additional audit checks include:

* 50/50 class balance;
* zero user overlap;
* zero pattern-tuple overlap;
* zero category-set overlap;
* negative-subtype balance constraints;
* train-derived time-normalization consistency;
* counterfactual validity checks.

The final evaluation uses frozen checkpoints and validation-derived operating thresholds.

Exact commands should remain synchronized with the current repository implementation. This README intentionally avoids duplicating command lines that may become stale as the research pipeline evolves.

---

# Portfolio-level summary

> Developed an independent behavioral-sequence ML research pipeline for Socia to study recurrence detection on a controlled synthetic longitudinal benchmark. Redesigned and audited the benchmark after identifying shortcut risks in an initial version, then evaluated multiple Transformer variants, an aggregate timing-and-category baseline, recurrence-representation ablations, counterfactual interventions, subtype/error structure, recurrence-count behavior, and a zero-shot LLM reference. The experiments show that aggregate timing/category features remain highly competitive, while neural models exhibit substantial dependence on their recurrence-related representations. The research remains an offline experimental system evaluated on synthetic data and is not yet validated on real behavioral sequences or integrated into Socia's production pipeline.
