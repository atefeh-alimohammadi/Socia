# Project status

This is the single source of truth for what exists. Other documents link here instead of restating it.

Categories are kept strictly separate:

* **Implemented**: present in code and wired into a running flow.
* **Partial**: present, but missing something the intended design needs.
* **Missing**: intended, not present.
* **Deferred**: consciously postponed.
* **Research**: exists outside the production pipeline and is not validated.

## Implemented

| Capability                                        | Where                                                    | Notes                                                                                               |
| ------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Registration, login, JWT auth                     | `api/auth.py`, `api/deps.py`                             | Login by email or username                                                                          |
| Onboarding (5 questions to memories, LLM welcome) | `api/onboarding.py`, `core/onboarding.py`                | Answers stored as `preference`/`insight` memories; deterministic fallback text if the LLM fails     |
| Memory-augmented chat                             | `api/conversation.py`, `ai/ollama.py`                    | Local Qwen2.5-7B                                                                                    |
| Deterministic self-harm check                     | `services/safety_service.py`                             | 10 regexes; fixed reply; see [SAFETY](SAFETY.md)                                                    |
| Message routing                                   | `services/orchestrator.py`                               | Exact-match skip list for trivial replies                                                           |
| Response consistency check                        | `services/reflection_service.py`                         | LLM judge, one regeneration, fails open                                                             |
| Signal extraction                                 | `services/conversation_analyzer.py`                      | 8 emotions (with valence/arousal), 7 behavior tags, 3 entity types                                  |
| Episodic memory + vector retrieval                | `services/episodic_memory_service.py`                    | pgvector, top 5, cosine distance <= 0.45                                                            |
| Pattern promotion                                 | `services/memory_service.py`                             | `count >= 3` per `(type, tag)`; versioned; evidence-linked                                          |
| Co-occurrence entity graph                        | `services/knowledge_graph_service.py`                    | Entities linked to tags per message                                                                 |
| Journeys from a pattern or from user text         | `api/journey.py`, `services/journey_service.py`          | 14–30 days; one challenge at a time                                                                 |
| Challenge feedback capture                        | `models/journey_challenge.py`                            | Difficulty (3 values), emotional response (free text), skip reason (8 values)                       |
| Message feedback                                  | `api/feedback.py`                                        | helpful / not_helpful                                                                               |
| Pacing policy                                     | `services/personalization_service.py`                    | Deterministic; affects chat tone only                                                               |
| Analytics                                         | `api/analytics.py`                                       | 8-week emotion timeline, valence/arousal trend, patterns, related entities                          |
| Journal with emotion analysis                     | `api/journal.py`, `services/emotion_analysis.py`         | Standalone; see Partial                                                                             |
| Per-user authorization                            | `services/authorization.py`, per-query `user_id` filters |                                                                                                     |
| Production logging without user message content   | Production logging paths                                 | User message content is not written to application logs                                             |
| Atomic Journey state transitions                  | `api/journey.py`                                         | Journey creation and challenge advancement commit only after required challenge generation succeeds |
| Conversation context limit                        | `api/conversation.py`                                    | Latest 20 messages are used for model context                                                       |
| Configurable LLM and application settings         | `core/config.py`, `ai/ollama.py`                         | Ollama URL, model name, embedding model, and CORS origin are environment-configurable               |
| Conversation-derived data deletion                | `models/*`, Alembic migrations                           | Conversation-dependent derived rows use appropriate cascading deletes                               |
| Evidence-based pattern synthesis trigger          | `services/conversation_analyzer.py`                      | Triggered by newly created behavior observations rather than global count divisibility              |

## Partial

| Capability                     | What exists                                                         | What is missing                                                                            |
| ------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Human-in-the-loop for patterns | Users can choose whether to start a Journey from a detected pattern | No way to dismiss, correct, or snooze an inferred pattern                                  |
| Adaptive planning              | The next challenge is conditioned on the previous challenge outcome | No persistent goal/plan model; adaptation does not consider the full Journey history       |
| Safety                         | Deterministic self-harm detection for chat messages                 | Other risk categories and input channels are not covered; no user-facing safety disclaimer |
| Journal                        | Journal CRUD with LLM-based emotion analysis                        | Journal entries do not currently feed into episodic memory or behavioral pattern detection |

## Missing

* Pattern dismiss / correction / snooze actions.
* Time-aware recurrence logic for behavioral patterns.
* Evaluation of signal extraction against a labeled dataset.
* Account deletion and data export.

## Deferred (intentional)

* Integrating the validated ML behavioral detector into the Production pattern-detection pipeline.
* Streaming responses, multi-device synchronization, and notifications.
* Multi-language support.

## Research (not integrated)

The Behavioral Detection Research Module in `ml/` is independent of the Production pipeline and is currently under validation. No Production behavior depends on its experimental results.

The current Production pattern mechanism is a simple baseline:

**LLM signal extraction → deterministic recurrence/counting rule**

The intended future architecture is:

**LLM signal extraction → ML behavioral pattern detector → pattern/evidence layer → user-controlled Journeys**

The ML detector is not currently integrated into Production, and its experimental results are not Production performance claims.

See [`research/README.md`](research/README.md).

## Known issues

These are the remaining limitations that materially affect the current Production prototype.

| Issue                                                                             | Where                               | Impact                                                                         |
| --------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------ |
| Pattern confidence is an uncalibrated mean of LLM self-reported confidence values | `services/memory_service.py`        | Confidence should not be interpreted as a statistically calibrated probability |
| Pattern recurrence is not time-aware                                              | `services/memory_service.py`        | Older evidence is treated similarly to recent evidence                         |
| Signal extraction has not been evaluated against a labeled dataset                | `services/conversation_analyzer.py` | Production extraction quality is not quantitatively established                |
| Safety coverage is limited to the implemented self-harm detection path            | `services/safety_service.py`        | Other risk categories and input channels are not currently covered             |
| Background analysis runs in-process via `BackgroundTasks`                         | `api/conversation.py`               | Analysis can be lost on process restart and has no durable retry mechanism     |
| JWT tokens are stored in `localStorage`                                           | `frontend`                          | A compromised frontend could expose authentication tokens                      |
