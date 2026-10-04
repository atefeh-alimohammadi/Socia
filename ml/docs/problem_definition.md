# Problem Definition

## Context

Socia is a conversational companion system with two conceptually separate
parts:

* **Production core**: an LLM extracts structured behavioral events from
  conversation. These observations feed memory, Journeys, and reflection/
  orchestration. A deterministic application rule currently promotes a
  repeated observation group to a user-visible "pattern" after **3 or more**
  **observations**.
* **Research core** (`ml/`): an independent offline investigation into
  whether recurring structure can be detected from temporal behavioral-event
  sequences, and how different modeling approaches compare on that task.

These two parts are **not integrated yet**.

The research models currently run only as offline experiments. They do not
run in the production application, do not influence the current
3+-observation promotion rule, and do not generate user-facing patterns.

The intended future direction is to evaluate the research models further and,
if they meet the required validation criteria, connect a suitable model to
the production pattern-detection pipeline.

The research benchmark is also deliberately narrower than Socia's
production notion of a "behavioral pattern": it defines recurrence using
explicit temporal and category-level rules in a synthetic dataset.

See [Research vs. Production boundary](#research-vs-production-boundary)
below.

---

## Research question

The research asks:

> **Given a 25-event window from a longitudinal behavioral-event stream, can**
> **a model distinguish an operationally defined recurring pattern from**
> **sequences that are superficially similar in timing, category composition,**
> **or event order but do not satisfy the benchmark's recurrence definition?**

A second question is used to place the learned models in context:

> **How does this recurrence-detection task compare across aggregate**
> **feature-based models, supervised temporal sequence models, and a**
> **general-purpose LLM used zero-shot?**

This second comparison is important because good performance from a neural
sequence model alone would not establish that the task requires learned
sequence reasoning. A simpler timing + category baseline may capture much of
the label signal, while a general-purpose LLM provides a different reference
point: whether a capable language model can perform the task directly from
the provided event sequence without task-specific supervised training.

The project therefore evaluates three broad approaches:

1. **Aggregate feature baseline** — a timing + category gradient-boosting
   model without explicit sequence modeling.
2. **Supervised sequence models** — the V2/V3 Transformer variants trained
   specifically for the benchmark.
3. **Zero-shot LLM reference** — Qwen2.5-7B evaluated with a fixed prompt
   on the same frozen V2 test benchmark, without task-specific fine-tuning.

The LLM experiment is a **single-model reference experiment**, not a claim
about the capabilities of LLMs in general.

The project initially used a broader formulation and a first-generation
benchmark (V1). A structured audit showed that V1 contained several
shortcuts through which a model could obtain strong performance without
requiring the recurrence reasoning implied by the original framing.

The benchmark was therefore redesigned as V2 to make recurrence more
explicitly controlled and to introduce targeted negative examples.

The current research should consequently be understood as an investigation
of **recurrence detection under a controlled synthetic definition**, with
comparisons across different modeling approaches—not as a validated
behavioral detector.

---

## What "recurrence" means in this benchmark

Recurrence is an **operational dataset definition**. It is not a
psychological or clinical definition of behavioral recurrence.

A sequence is labeled **positive** if and only if it contains at least
two temporally separated occurrences ("sites") of the same ordered
category tuple.

Each tuple contains **2–4 categories**.

For an occurrence/site to be valid:

1. Its categories must appear in the canonical order defined by the
   generator.
2. All events belonging to that occurrence must fall within a **24-hour**
   **burst span**.

For consecutive recurrence sites, the temporal separation must satisfy
both:

* an absolute gap of at least **2 days**; and
* a gap of at least **3× the burst span** of that particular pair.

The second condition makes the separation scale-relative rather than
depending only on a fixed global time window.

Formally, if two consecutive sites have burst span (B_i) and temporal
gap (G_i), they satisfy the recurrence-separation condition when:

$$\
G_i \geq 2\text{ days}\
$$

and

$$\
G_i \geq 3B_i\
$$

This definition is enforced by the dataset generator rather than inferred
after model evaluation.

---

## What is explicitly not recurrence?

Several boundary cases are deliberately excluded from the positive class.

### Single occurrence

A sequence containing exactly one well-formed occurrence of the tuple is
not positive.

These examples are represented by:

`boundary_single_occurrence`

### Tight burst

Two occurrences that are too close together to satisfy the persistence
condition are not considered temporally recurring under this benchmark.

These examples are represented by:

`boundary_tight_burst`

This distinction is important because repeated events within a single dense
episode are not equivalent to temporally separated recurrence under the
benchmark definition.

Other negative subtypes test different potential shortcuts; see
[`dataset.md`](dataset.md).

---

## Why recurrence is difficult to isolate

A sequence can contain information that correlates with a recurrence label
without actually containing the recurrence structure defined above.

The benchmark therefore attempts to separate several potentially
confounding signals.

### Timing and density

A sequence containing a dense burst of structured events can have timing
statistics that resemble a recurrence-positive example without containing
multiple temporally separated sites.

### Category composition

A window can contain the same categories as a recurring pattern without
containing repeated occurrences of that pattern.

This creates a distinction between:

* category presence/counts; and
* recurrence of an ordered category tuple.

### Frequency

A category occurring many times does not necessarily mean that a
multi-category pattern recurs.

For example, repeated instances of one category are not equivalent to
repeated sites of the same ordered tuple.

### Event order

The same category composition and similar timing can be arranged
differently.

A permutation of the event order is therefore treated separately from a
valid recurrence under the benchmark definition.

These distinctions motivate both the hard-negative construction and the
comparison between simple aggregate models and sequence models.

---

## Why the benchmark was redesigned

The original V1 benchmark was intentionally exploratory.

Its subsequent audit identified several sources of shortcut information,
including:

* approximately 54.6% of positive windows containing only one intended
  recurrence;
* substantial negative examples originating from a separate control-user
  population;
* strong performance from aggregate category/timing features;
* extensive overlap between selected windows;
* category-set relationships that could provide shortcut information.

These findings meant that good V1 performance could not be interpreted
cleanly as evidence of recurrence detection.

V2 was therefore redesigned to address these issues through:

* explicit multi-site positive construction;
* temporally separated recurrence sites;
* dedicated boundary negatives;
* multiple hard-negative subtypes;
* zero user overlap across splits;
* zero pattern-tuple overlap across splits;
* zero category-set overlap across splits;
* balanced train/validation/test classes;
* controlled negative-subtype proportions.

The V2 redesign should be described as **reducing identified benchmark**
**shortcuts**, not as proving that all possible shortcuts have been
eliminated.

See [`experiment_history.md`](experiment_history.md) and
[`dataset.md`](dataset.md).

---

## Negative classes

The current V2 benchmark contains six deliberately constructed negative
subtypes:

| Subtype                      | Purpose                                                      |
| ---------------------------- | ------------------------------------------------------------ |
| `timing_matched`             | Tests whether similar timing alone is sufficient             |
| `order_permutation`          | Tests sequences with altered internal order                  |
| `boundary_single_occurrence` | Separates one valid occurrence from recurrence               |
| `category_identity`          | Tests category-composition / identity shortcuts              |
| `boundary_tight_burst`       | Separates dense bursts from temporally persistent recurrence |
| `pure_background`            | Provides non-pattern background sequences                    |

These negative types are not intended to represent every possible
non-recurring sequence. They are controlled benchmark interventions
designed around the specific shortcut hypotheses being studied.

---

## Experimental comparison

The V2 benchmark is used to compare several modeling approaches under a
common frozen test protocol.

### 1. Timing + category baseline

A gradient-boosting classifier uses aggregate timing and category features,
including duration, gap statistics, category counts, unique-category
statistics, and entropy.

It does **not** model the event sequence directly.

This baseline answers an important question:

> How much of the benchmark can be solved from aggregate timing and
> category information without explicit sequence modeling?

### 2. Supervised temporal sequence models

The research evaluates several progressively modified Transformer models:

* **V2** — temporal-relation Transformer;
* **V3-A** — V2-style recurrence representation plus learned event-order
  embedding;
* **V3-C** — controlled V3-A ablation without the learned event-order
  embedding;
* **V3-A Bucketed** — V3-A using discrete recurrence-gap buckets.

These models are trained specifically on the synthetic benchmark.

### 3. Zero-shot LLM reference

To test whether the task can be solved directly by a general-purpose
language model without task-specific supervised training, **Qwen2.5-7B**
was evaluated zero-shot on the **same frozen V2 test benchmark**.

The experiment uses one fixed prompt and does not fine-tune the model on the
benchmark.

This experiment provides a reference point outside the supervised
sequence-model family:

> Can a general-purpose 7B language model classify these temporal event
> sequences directly from their presented structure?

The result is interpreted narrowly as evidence about **this model, this**
**prompt, and this benchmark protocol**. It is not generalized to LLMs as a
class.

---

## What the research is actually evaluating

The research therefore evaluates several different questions rather than
treating one metric as proof of recurrence understanding.

### Overall discrimination

ROC-AUC and PR-AUC measure how well each approach ranks positive and
negative windows across thresholds.

### Classification at a validation-selected threshold

F1, precision, recall, and accuracy are reported at a threshold selected
using the validation set and then frozen for the test set.

### Dependence on recurrence representations

Ablation and permutation experiments test whether predictions change when
recurrence-related representations are removed or disrupted.

### Sensitivity to controlled counterfactual changes

Counterfactual experiments modify recurrence-related temporal information
while preserving other aspects of an example as much as possible.

### Error structure

Negative-subtype analysis examines which controlled confounds are most
frequently mistaken for recurrence.

### Recurrence-count behavior

The analysis checks whether positive recall changes systematically with
the number of intended recurrence sites.

### General-purpose LLM reference

The zero-shot Qwen2.5-7B experiment provides an additional comparison
against a general-purpose language model that was not trained specifically
for this benchmark.

Together, these experiments provide a more informative picture than a
single test-set F1 score.

---

## Interpreting the model comparisons

The comparison is intentionally not framed as:

> "Which model understands behavior best?"

Instead, it asks what each approach can accomplish under the synthetic
operational definition.

A neural model performing well does not by itself demonstrate that it has
learned recurrence-specific structure. Conversely, a simple baseline
performing strongly is useful evidence that aggregate benchmark features
carry substantial predictive information.

The zero-shot LLM result answers a different question: whether a
general-purpose language model can perform this classification directly
without task-specific supervised training.

Because the LLM experiment uses one model and one prompt, it should not be
used to conclude that LLMs are generally unsuitable for this task.

---

## Research interpretation boundary

The benchmark allows us to ask whether model predictions change when
recurrence-related inputs are removed or disrupted.

For example, the current ablation experiments show substantial drops in
ranking performance when recurrence-related representations are replaced
or globally permuted.

This provides evidence that the trained models **depend on those**
**representations**.

However, this does not establish that the models have learned:

* a psychologically validated concept of behavioral recurrence;
* a causal representation of recurrence;
* semantic understanding of human behavior;
* a general-purpose behavioral pattern detector.

The synthetic benchmark itself defines what counts as recurrence.

Therefore the strongest supported interpretation is:

> The supervised models learn to use recurrence-related representations
> that are predictive under the controlled synthetic benchmark, while
> real-world behavioral validity and generalization remain unestablished.

The zero-shot Qwen2.5-7B result should be interpreted separately: it is a
single reference experiment showing how one general-purpose LLM performed
under the chosen zero-shot protocol.

---

## Explicit non-goals

This research module does **not**:

* diagnose or claim to detect social anxiety, depression, ADHD, or any
  other clinical or psychological condition;
* claim to detect or understand real human behavior;
* establish that a user's real-world behavior follows the synthetic
  recurrence definition;
* claim that the current models are clinically validated;
* claim that the neural models outperform the timing + category baseline;
* claim that recurrence-feature ablations prove causal understanding;
* claim that one Qwen2.5-7B zero-shot result represents LLMs generally;
* represent, feed, or influence Socia's production pattern-detection rule;
* establish that the research benchmark is a valid proxy for real-world
  longitudinal behavior.

---

## Research vs. Production boundary

The production and research systems share a conceptual motivation but are
technically separate.

```mermaid
flowchart TB

    subgraph PROD["Production system — live application"]
        P1["LLM behavioral-event<br/>extraction"]
        P2["Deterministic promotion rule:<br/>3+ observations → pattern"]
        P3["Memory / Journeys /<br/>Reflection / Orchestration"]

        P1 --> P2
        P2 --> P3
    end

    subgraph RESEARCH["Research module — offline"]
        R1["V2 synthetic benchmark"]
        R2["Timing + category baseline"]
        R3["V2 / V3-A / V3-C /<br/>V3-A Bucketed"]
        R4["Ablation + counterfactual<br/>evaluation"]
        R5["Qwen2.5-7B<br/>zero-shot reference"]

        R1 --> R2
        R1 --> R3
        R3 --> R4
        R1 --> R5
    end

    PROD -.->|"No runtime connection"| RESEARCH
```

### Component boundaries

| Component                        | Function                                                      | Environment      |
| -------------------------------- | ------------------------------------------------------------- | ---------------- |
| LLM signal extraction            | Converts conversation into structured behavioral observations | Production       |
| Deterministic 3+ rule            | Promotes repeated observations to a production pattern        | Production       |
| V2 synthetic benchmark           | Defines and generates the controlled recurrence task          | Research         |
| Timing + category baseline       | Measures performance from aggregate features                  | Offline research |
| V2 / V3-A / V3-C                 | Sequence-model experiments                                    | Offline research |
| V3-A Bucketed                    | Discrete recurrence-representation experiment                 | Offline research |
| Ablation/counterfactual analysis | Tests model dependence and output sensitivity                 | Offline research |
| Qwen2.5-7B benchmark             | One zero-shot LLM reference experiment                        | Offline research |

The research module currently has **no effect on production predictions,**
**memory, Journeys, or user-visible pattern generation**.

---

## Relationship to the production 3+ rule

The production rule and the research benchmark solve related but different
problems.

The production rule answers a simple application question:

> Has this observation been seen at least three times?

The research benchmark asks a more structured sequence question:

> Does this 25-event window contain at least two temporally separated
> occurrences of the same ordered category tuple under the benchmark's
> recurrence definition?

Therefore the research benchmark should not be described as a direct
evaluation of whether the production rule is "wrong" or "correct."

Instead, the fixed-count rule provides the practical motivation for
investigating whether a more sequence-aware approach could eventually
represent recurrence more explicitly.

The research comparison then asks whether that more structured task is
better served by aggregate features, supervised temporal models, or a
general-purpose zero-shot LLM.

---

## Current status

The research module is an **offline proof-of-concept / experimental**
**benchmarking pipeline**.

The current evidence establishes that:

* the V2 benchmark is reproducible and audited against several known
  leakage/shortcut risks;
* a simple timing + category baseline remains highly competitive;
* the supervised neural models provide a controlled comparison of several
  temporal representations but do not establish superiority over the
  aggregate baseline;
* the trained neural models exhibit substantial dependence on their
  recurrence-related representations;
* the current counterfactual experiments do not establish reliable
  order sensitivity;
* the zero-shot Qwen2.5-7B experiment provides a useful single-model
  reference, but is not evidence about LLMs generally.

It does **not** establish real-world behavioral validity.

Future integration into Socia would require additional validation,
particularly on appropriate real longitudinal data, before any research
model could reasonably be considered for production use.
