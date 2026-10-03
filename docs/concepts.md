# Concepts and glossary

| Term | Meaning in Socia | Where |
|---|---|---|
| **Signal** | Something extracted from a message by the LLM under a fixed schema: an emotion, a behavior observation, or an entity | `conversation_analyzer.py` |
| **Emotion** | One of 8 labels (anxiety, confidence, fear, excitement, frustration, sadness, shame, hope) with intensity, plus estimated valence (-1..1) and arousal (0..1). Estimates, not diagnoses | `conversation_emotions` |
| **Behavior tag** | One of 7 labels: fear_of_judgment, avoiding_social_expression, overthinking, low_confidence, social_avoidance, seeking_validation, conflict_avoidance | `behavior_observations` |
| **Entity** | A short lowercase name typed as person, situation or topic (e.g. "presentation") | `knowledge_entities` |
| **Episodic memory** | One specific moment: a single extracted emotion or behavior with its evidence sentence and embedding. One occasion, not a trend | `episodic_memory` |
| **Synthesized memory** | A stable statement about the user: a *pattern*, *emotion pattern*, or an onboarding *preference/insight* | `user_memory` |
| **Pattern** | A behavior tag that recurred at least 3 times, promoted to a synthesized memory with a one-sentence description. Currently produced by a counting rule, not a trained detector | `memory_service.py` |
| **Pulse** | The UI's name for a pattern. A Journey created from one shows "Associated Pulse #id". Same object as a pattern | `frontend/app/journeys/[id]` |
| **Evidence link** | Connects a synthesized memory to the episodic rows cited for it | `user_memory_evidence` |
| **Journey** | A 14-30 day process of working on one pattern or user-stated goal | `journeys` |
| **Challenge** | One small task inside a Journey, generated after the previous one is resolved | `journey_challenges` |
| **Pacing policy** | `gentle`, `direct` or `standard`, derived deterministically from recent feedback; changes the chat tone instruction | `personalization_service.py` |
| **Reflection** | An LLM judge checking a reply for consistency with known user context, with one regeneration | `reflection_service.py` |
| **Candidate (intended)** | The intended status of a pattern before the user validates it. Not yet a stored state; see [status](status.md) | n/a |

Two distinctions are deliberate and used consistently:

- **Episodic vs synthesized.** The system prompt tells the model that episodic items are single moments and only synthesized items may be described as ongoing tendencies.
- **Detected vs validated.** Nothing Socia detects is treated as validated truth about the user. Detection quality is unmeasured (see [status](status.md)).
