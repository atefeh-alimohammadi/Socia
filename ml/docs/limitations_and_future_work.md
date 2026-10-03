# Limitations and Future Work

## Limitations

These limitations define the scientific scope of the current study. They are
important when interpreting both the benchmark results and the model
ablations.

### Synthetic data only

The benchmark is entirely synthetic. No real longitudinal user data was used,
referenced, or claimed to be represented.

The generator uses controlled assumptions, including a fixed 15-category
vocabulary, synthetic background events, and designed recurrence timing.
These choices make controlled experimentation possible, but they should not be
interpreted as a statistical model of real human behavior.

### Unknown synthetic-to-real gap

The experiments establish model behavior on the constructed benchmark, not on
real conversational or behavioral sequences.

In particular, the study does not establish that recurrence in real
behavioral-event streams has the same temporal structure, category
distribution, background rate, or noise characteristics as recurrence in the
synthetic data.

Understanding and measuring this synthetic-to-real gap would require an
appropriate real or independently constructed longitudinal dataset.

### Neural models do not outperform the aggregate baseline

On the final V2 benchmark, the timing+category gradient-boosting baseline
remains competitive with, and on the primary ranking metrics outperforms, the
neural models.

The baseline reaches approximately:

* ROC-AUC: **0.807**
* PR-AUC: **0.759**

compared with:

* V2: ROC-AUC **0.747**, PR-AUC **0.701**
* V3-A: ROC-AUC **0.774**, PR-AUC **0.722**
* V3-C: ROC-AUC **0.770**, PR-AUC **0.716**
* V3-A Bucketed: ROC-AUC **0.776**, PR-AUC **0.730**

The neural models therefore should not currently be presented as superior
detectors of recurrence on this benchmark.

### Recurrence-representation dependence is not the same as recurrence understanding

The ablation experiments show that the neural models depend substantially on
their recurrence-related input representation.

For example, replacing the continuous recurrence representation with a
no-prior representation reduced ROC-AUC from:

* **0.747 → 0.401** for V2
* **0.774 → 0.652** for V3-A
* **0.770 → 0.653** for V3-C

Global permutation of recurrence-feature pairs also substantially degraded
ranking performance.

These results demonstrate sensitivity to the supplied recurrence
representation and, in the permutation experiment, to its event-level
alignment.

They do **not** establish that the models have learned a psychologically
meaningful or causally valid concept of behavioral recurrence. The
interventions are model-level diagnostics on a synthetic benchmark.

### Order sensitivity remains unresolved

The order-permutation counterfactual did not produce a strong or consistent
expected-direction response from the neural models.

Across the evaluated positive examples, the fraction of cases moving in the
expected direction was approximately 48–53% for the continuous V3 variants.

This is close to chance and does not support a strong claim that the learned
order embedding or relative-time attention mechanism produces reliable
order-sensitive predictions under the current evaluation.

This should be interpreted as an unresolved capability rather than proof that
the architectures cannot represent order.

### Residual timing and category shortcuts remain possible

The redesigned benchmark was constructed to reduce several shortcut
opportunities, but it does not eliminate all information available through
aggregate timing and category statistics.

The timing+category baseline achieving approximately **0.81 ROC-AUC** is direct
evidence that substantial predictive information remains accessible without a
sequence model.

Consequently, high performance on this benchmark cannot automatically be
interpreted as evidence that a model has recovered the intended recurrence
structure.

### Benchmark-specific recurrence definition

The positive class is defined using a specific operational construction:
repeated sites of the same ordered category tuple with temporal separation
constraints.

This definition is useful for controlled experiments, but it is narrower than
the broader concept of recurrence that might arise in real behavioral data.

Different recurrence definitions could produce different model rankings and
failure modes.

### Limited LLM comparison

The LLM benchmark evaluates only **Qwen2.5-7B**, zero-shot, using one fixed
prompt.

The result therefore represents a single model-and-prompt configuration. It
does not characterize LLMs as a class or establish the limits of
instruction-following models on this task.

The observed tendency toward predicting almost all examples as positive may
reflect the specific prompt, model calibration, task formulation, or some
combination of these factors.

### Test-time interventions are diagnostic, not training results

The recurrence-feature ablations and counterfactual interventions are applied
to trained models without retraining.

They therefore measure sensitivity of an existing model to controlled input
changes. They do not measure how a model trained under those modified
conditions would learn the task.

Similarly, best-test-F1 thresholds are diagnostic because they use test labels;
they are not treated as primary generalization metrics.

### No clinical or psychological validation

The behavioral categories used by the benchmark, such as
`social_avoidance`, `overthinking`, or `fear_of_judgment`, belong to an
application-oriented synthetic taxonomy.

They are not clinical constructs, diagnostic categories, or validated
psychological measurements.

No clinical or psychological interpretation is claimed.

### No real-world behavioral detector

The research module is an offline experimental system rather than a validated
behavioral detector.

It does not establish that a person exhibits a particular behavioral trait,
and its predictions should not be interpreted as psychological assessment.

### Research/production separation

The research module is not currently integrated into Socia's production
pipeline.

The production system uses a separate deterministic rule that promotes
repeated extracted observations after the configured 3+ observation threshold.
The research benchmark defines recurrence differently and should not be
treated as a direct validation of that production rule.

Nothing in the current research evaluation establishes that the neural models
are ready to replace the production mechanism.

---

# Future Work

The following are potential directions motivated by the current evidence.
They are **not presented as completed work or as a fixed development
roadmap**.

## Near term

### Understand the remaining baseline advantage

A useful next experiment would be to determine why aggregate timing and
category statistics remain so competitive.

Possible questions include:

* How much performance remains when timing statistics are more tightly
  matched?
* How much remains when category-frequency statistics are controlled?
* Which negative subtypes contribute most to the baseline advantage?
* Does the neural model actually use sequence structure when aggregate cues
  are weakened?

This would help distinguish architectural limitations from benchmark
information that is still accessible through simple statistics.

### Isolate recurrence representation more carefully

The current ablations show strong dependence on recurrence-related features, but
they do not identify exactly which information is responsible.

Possible follow-up experiments include separately ablating:

* same-category recurrence interval;
* prior-occurrence indicator;
* absolute time;
* immediate-event gap;
* relative-time attention bias;
* recurrence buckets.

Such experiments could determine whether the observed sensitivity is driven by
one feature or by the interaction of several temporal representations.

### Investigate order sensitivity

The current counterfactual results do not provide strong evidence of reliable
order sensitivity.

A controlled follow-up could compare:

* explicit order supervision;
* reorder-verification auxiliary objectives;
* alternative positional representations;
* stronger order-permutation negatives.

Any such experiment should preserve the current benchmark split and retain
counterfactual evaluation so that improvements are not inferred solely from
aggregate test metrics.

### Strengthen the hardest negative constructions

The most informative next benchmark changes would target negative examples
that remain difficult for the models or where baseline and neural behavior
diverge substantially.

In particular, `timing_matched` and `order_permutation` provide useful targets
for further stress testing.

Any new negative subtype should undergo the same split-overlap and shortcut
auditing used for the current benchmark.

### Broaden the LLM reference

If compute and infrastructure permit, evaluating additional models and prompt
variants would make the LLM comparison more informative.

This should be treated as a separate reference experiment rather than as a
central part of the neural-model study.

---

## Medium term

### More realistic longitudinal simulation

A future benchmark could introduce richer temporal structure, such as:

* non-stationary background event rates;
* user-specific event distributions;
* multiple co-occurring behavioral patterns;
* variable recurrence intervals;
* heterogeneous pattern strengths;
* more realistic background noise.

The goal would not simply be to make the benchmark harder, but to make the
generative assumptions more representative while retaining a precisely known
ground truth.

### Stronger cross-pattern generalization tests

The current benchmark already separates pattern identities and category sets
across splits.

Further experiments could systematically test generalization across:

* unseen tuple lengths;
* unseen combinations of categories;
* different recurrence-gap distributions;
* different background rates;
* different pattern densities.

This would help establish whether performance depends on memorizing particular
synthetic pattern shapes.

### Alternative temporal architectures

Other temporal representations could be evaluated, including:

* explicit time embeddings;
* continuous-time representations such as Time2Vec;
* deeper or more expressive temporal encoders;
* architectures designed specifically for irregular event sequences.

Such comparisons would be most useful after the remaining benchmark shortcuts
are better characterized.

### Better-controlled model comparisons

The V2 and V3-family experiments use different training objectives, while
V3-A/V3-C form the cleanest architectural ablation.

A future model comparison should make the intended comparison axis explicit:
architecture, temporal representation, training objective, or some combination
of these.

This would reduce the risk of attributing performance differences to the wrong
component.

---

## Long term

### Real longitudinal validation

The largest unresolved step is evaluation on appropriately collected
longitudinal data, if such data becomes available and can be used ethically
and with appropriate privacy protections.

This would allow the study to test whether the synthetic recurrence definition
and model behavior transfer to real event streams.

### Synthetic-to-real transfer

Rather than assuming that a synthetic benchmark transfers to real data, future
work should explicitly measure the transfer gap.

Potential evaluation could compare performance under controlled shifts in:

* event frequency;
* category distributions;
* temporal dynamics;
* recurrence strength;
* user heterogeneity.

### Potential production integration

Only after substantial validation would it be meaningful to investigate whether
a validated ML recurrence detector could complement or replace Socia's current
deterministic 3+-observation rule.

Such integration would require evidence beyond the current synthetic benchmark,
including robustness, calibration, failure analysis, and appropriate
user-safety evaluation.

It is therefore a possible long-term direction, not a current claim or
near-term integration plan.
