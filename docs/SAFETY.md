# Safety, privacy and responsible use

Socia is designed as a self-reflection and practice companion for people who may experience difficulties with social communication. Because the system can produce statements about a user's behavior and emotions, the project prioritizes explicit limits, transparent failure modes, and user control.

## Scope statement

* Socia is a **self-reflection and practice companion prototype**. It is **not** a therapist, diagnostic tool, medical device, or crisis service.
* The system has **no clinical validation**. No claim is made that it improves wellbeing, treats a condition, or reduces social anxiety.
* Detected "patterns" are **application-level hypotheses synthesized from a language model's interpretation of the user's messages**. Their accuracy has not been measured and they can be wrong.
* If someone is in immediate danger, experiencing a crisis, or considering self-harm, Socia should not be relied on as the sole source of support. They should contact appropriate local emergency services, a crisis service, or a trusted person.

## Safeguards currently implemented

| Safeguard                       | Mechanism                                                                                                                                                                            | Limits                                                                                                                                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Self-harm keyword short-circuit | A small set of case-insensitive regex rules runs before model calls for chat messages. On a match, the system returns a fixed supportive response instead of an LLM-generated reply. | Keyword-based rather than comprehensive risk detection; can miss paraphrases and other risk categories. Message text is currently logged, and no region-specific resource list is attached. |
| Non-therapist framing           | System prompts state that Socia is not a therapist and should avoid medical or legal advice.                                                                                         | Prompt-level only; not independently enforced.                                                                                                                                              |
| Output consistency check        | A separate LLM check looks for contradictions and ignored distress and can trigger one regeneration.                                                                                 | This is a consistency mechanism, not a safety classifier or guarantee. It can fail open and only evaluates chat replies.                                                                    |
| Anti-diagnosis prompting        | Prompts distinguish individual observations from synthesized patterns and instruct the model not to diagnose or use past messages as proof.                                          | Prompt-level only; model behavior is not guaranteed.                                                                                                                                        |
| Gentle pattern wording          | Synthesized pattern descriptions are requested to be supportive, non-clinical, and phrased as observations rather than diagnoses.                                                    | Generated text can still be inaccurate or misleading.                                                                                                                                       |
| User-controlled Journeys        | Creating or reactivating a Journey requires an explicit user action.                                                                                                                 | Patterns cannot currently be dismissed, corrected, or marked as "not me".                                                                                                                   |
| Pacing adaptation               | Recent feedback can adjust conversational pacing toward `gentle`, `standard`, or `direct`.                                                                                           | This affects chat tone only and does not independently evaluate user safety or challenge suitability.                                                                                       |
| Per-user data scoping           | Application queries scope user-owned data to the authenticated user.                                                                                                                 | This is an application-level isolation mechanism, not a complete security guarantee.                                                                                                        |

## Known safety gaps

These limitations are documented so that the prototype is not mistaken for a validated safety system.

* The current self-harm check applies to chat messages only. It does not run across every free-text input, such as onboarding answers, journal entries, or challenge feedback.
* There is no dedicated detection for other risk categories such as harm to others, abuse, or eating disorders.
* There is no output-side safety classifier. A generated reply or challenge can therefore still be unsafe, inappropriate, or factually wrong.
* The reviewed frontend does not currently display a persistent "not a therapist / not for emergencies" notice.
* **Emotional reliance is not monitored.** There are no usage limits, dependence-oriented check-ins, or systematic prompts encouraging human contact.
* **Hallucination remains possible.** The model can state things about the user that are not supported by stored memory. The current consistency mechanism provides only one additional model-based check.
* **Pattern mislabeling is possible.** A pattern displayed with an observation count and model-derived percentage can appear more authoritative than warranted. The displayed confidence is an uncalibrated aggregation of model self-reported confidence values.
* Challenges are generated by the LLM. The prompt asks for small, generic, achievable activities, but there is currently no independent suitability or risk review of generated challenges.

## Privacy and data handling

### Where data goes

The current production code path uses a **locally hosted Ollama model** for chat, extraction, reflection, and Journey generation. Under this code path, message content is not sent to a third-party LLM API.

A legacy Gemini adapter and related configuration remain in the codebase but are not used by the current path.

"Local" here describes the current model-serving path; it does not by itself guarantee that the application is secure or that stored data is protected from unauthorized access.

### What is stored

The PostgreSQL database currently stores, without application-level field encryption:

* chat messages;
* emotions, including intensity, valence, and arousal;
* behavior observations and extracted evidence sentences;
* embeddings;
* synthesized memories and their history;
* entities and entity edges;
* Journeys and challenge feedback;
* free-text emotional responses;
* journal entries.

The application should therefore be treated as handling sensitive user-generated data even though the current model inference path is local.

### Known privacy issues

The following should be addressed before production deployment with real user data:

| Issue                                  | Detail                                                                                                                                                                                                                                                                            |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Message content in logs                | The safety check currently logs each message at `WARNING`; retrieval-miss logging can include the query at `INFO`; challenge-context logging can include previous challenge feedback at `INFO`.                                                                                   |
| Token in `localStorage`                | Storing the authentication token in `localStorage` increases exposure in the event of an XSS vulnerability.                                                                                                                                                                       |
| No retention policy                    | Stored data currently has no defined automatic retention period.                                                                                                                                                                                                                  |
| No account-level deletion or export    | The current API provides some per-resource deletion but no complete account deletion or data-export workflow.                                                                                                                                                                     |
| Conversation deletion and derived data | Deleting a conversation may interact with foreign-key relationships from derived records; derived memories are not currently designed around automatic deletion with their source conversation. This behavior should be verified before relying on deletion for complete erasure. |

## Threat notes

### Prompt injection and self-injection

User text is provided to extraction, reflection, and pattern-description prompts. Structured outputs are constrained by closed vocabularies and database validation, which limits some forms of structured injection.

However, free-text evidence sentences are stored and can later re-enter the same user's system prompt. This creates a **within-user feedback path**: previously extracted content can influence later retrieval, extraction, and responses.

Retrieval is scoped by `user_id`, so this mechanism is not intended to expose one user's memories to another user. Delimiting and sanitizing stored evidence text remains a planned hardening step.

### Authorization

Ownership checks are applied at the resource level. Invalid or unauthorized resource IDs return an error rather than exposing the requested resource.

### Availability

LLM calls are performed synchronously in request handlers and there is currently no application-level rate limiting. A deployment can therefore be susceptible to resource exhaustion through repeated requests or expensive model calls.

## Recommendations before production use

1. Remove user-generated content from application logs and establish a documented retention and deletion policy.
2. Add a clear frontend notice describing the system's scope and provide appropriate crisis resources.
3. Extend risk handling across all relevant free-text inputs and broaden the categories covered.
4. Avoid presenting uncalibrated model confidence as a probability; prefer transparent signals such as evidence count and recency.
5. Add user controls to dismiss, correct, or reject synthesized patterns.
6. Evaluate extraction accuracy on a labeled dataset before presenting synthesized patterns as meaningful findings.
7. Add independent suitability and safety checks for generated challenges.
8. Add complete account deletion and data export, and verify cascading deletion of derived data.
9. Address authentication-token storage, logging, rate limiting, and other application-security concerns before handling real user data at scale.
10. Any claim about clinical or wellbeing outcomes would require appropriate validation and oversight. No such claim is currently made.

## Reporting a concern

<!-- TODO: add a security/contact channel, such as a project email or GitHub Security Advisory. -->
