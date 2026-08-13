
import json
import logging

from sqlalchemy.orm import Session

from app.ai.ollama import generate_ai_response
from app.database.database import SessionLocal

from app.models.conversation_emotion import ConversationEmotion
from app.models.behavior_observation import BehaviorObservation
from app.models.knowledge_entity import KnowledgeEntity
from app.models.knowledge_edge import KnowledgeEdge

from app.services.memory_service import (
    synthesize_patterns_from_observations,
)
from app.services.episodic_memory_service import (
    create_episodic_memory,
)

logger = logging.getLogger(__name__)


VALID_EMOTIONS = {
    "anxiety",
    "confidence",
    "fear",
    "excitement",
    "frustration",
    "sadness",
    "shame",
    "hope",
}


VALID_TAGS = {
    "fear_of_judgment",
    "avoiding_social_expression",
    "overthinking",
    "low_confidence",
    "social_avoidance",
    "seeking_validation",
    "conflict_avoidance",
}


VALID_ENTITY_TYPES = {
    "person",
    "situation",
    "topic",
}


def build_analyzer_prompt(content: str):

    return f"""
You are analyzing a user's conversation message for social behavior signals.

User message:

"{content}"

Extract emotions, behavior observations, and simple entities.

The user's message may describe multiple distinct events or topics. For
EACH emotion and EACH behavior observation you extract, the "evidence"
text must be specific to that one event only - do not summarize or repeat
the entire message. Quote or closely paraphrase only the part of the
message relevant to that specific emotion or behavior.

For entities, extract only clearly mentioned concrete entities that are
useful for understanding recurring patterns in the user's life.

Entity types:

- person: a person or group of people relevant to the message
  Examples: "my friend", "my manager", "my parents"

- situation: a situation, activity, or circumstance
  Examples: "presentation", "group discussion", "job interview"

- topic: a subject, object, or recurring topic
  Examples: "Python", "work", "university"

Return ONLY valid JSON:

{{
"emotions":[
{{
"emotion":"anxiety",
"intensity":0.8,
"evidence":"I kept worrying that I would forget everything during the presentation."
}}
],
"observations":[
{{
"tag":"fear_of_judgment",
"evidence":"I was terrified that everyone would think my presentation was terrible.",
"confidence":0.85
}}
],
"entities":[
{{
"name":"presentation",
"entity_type":"situation"
}}
]
}}

Valid emotions:
anxiety, confidence, fear, excitement,
frustration, sadness, shame, hope

Valid behavior tags:
fear_of_judgment,
avoiding_social_expression,
overthinking,
low_confidence,
social_avoidance,
seeking_validation,
conflict_avoidance

Valid entity types:
person,
situation,
topic

Rules:

- Only extract clearly present signals.
- Do not invent patterns.
- Intensity and confidence must be between 0 and 1.
- Evidence must be one sentence, grounded in the specific part of the
  message relevant to that emotion/behavior.
- Do not use generic restatements of the tag or emotion name.
- If the message contains multiple distinct events, each extracted
  emotion/observation must point to its own specific evidence.
- Entities must be explicitly mentioned or clearly identifiable from
  the message.
- Do not invent entities.
- Entity names should be short and normalized.
- Use lowercase entity names.
- If no useful entities are present, return an empty entities list.
- Return empty lists when no signals exist.
- Return JSON only.
"""


def analyze_conversation_message(
    message_id: int,
    user_id: int,
    session_id: int,
    content: str,
) -> None:

    db = SessionLocal()

    try:

        prompt = build_analyzer_prompt(content)

        response = generate_ai_response(
            prompt
        )

        cleaned_response = (
            response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        data = json.loads(
            cleaned_response
        )

        if (
            "emotions" not in data
            or "observations" not in data
        ):
            raise ValueError(
                "Invalid analyzer response format"
            )

        emotions = data.get(
            "emotions",
            []
        )

        observations = data.get(
            "observations",
            []
        )

        entities = data.get(
            "entities",
            []
        )

        # ---------------------------------------------------------
        # Emotions
        # ---------------------------------------------------------

        extracted_memories = []

        for item in emotions:

            emotion_name = item.get(
                "emotion"
            )

            intensity = item.get(
                "intensity"
            )

            emotion_evidence = item.get(
                "evidence"
            )

            if emotion_name not in VALID_EMOTIONS:
                continue

            if not isinstance(
                intensity,
                (int, float)
            ):
                continue

            if intensity < 0 or intensity > 1:
                continue

            if not emotion_evidence:
                continue

            emotion = ConversationEmotion(
                message_id=message_id,
                user_id=user_id,
                emotion=emotion_name,
                intensity=intensity,
            )

            db.add(emotion)

            memory = create_episodic_memory(
                db,
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
                event_type="emotion",
                tag=emotion_name,
                content=f"{emotion_name}: {emotion_evidence}",
                confidence=intensity,
            )

            extracted_memories.append(
                (
                    memory,
                    emotion_name,
                )
            )

        # ---------------------------------------------------------
        # Behaviors
        # ---------------------------------------------------------

        for item in observations:

            tag = item.get(
                "tag"
            )

            evidence = item.get(
                "evidence"
            )

            confidence = item.get(
                "confidence"
            )

            if tag not in VALID_TAGS:
                continue

            if not evidence:
                continue

            if not isinstance(
                confidence,
                (int, float)
            ):
                continue

            if confidence < 0 or confidence > 1:
                continue

            observation = BehaviorObservation(
                user_id=user_id,
                message_id=message_id,
                session_id=session_id,
                tag=tag,
                evidence=evidence,
                confidence=confidence,
            )

            db.add(
                observation
            )

            memory = create_episodic_memory(
                db,
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
                event_type="behavior",
                tag=tag,
                content=f"{tag}: {evidence}",
                confidence=confidence,
            )

            extracted_memories.append(
                (
                    memory,
                    tag,
                )
            )

        # ---------------------------------------------------------
        # Make sure all episodic memories have IDs
        # before creating KnowledgeEdge rows.
        # ---------------------------------------------------------

        db.flush()

        # ---------------------------------------------------------
        # Knowledge Graph
        # ---------------------------------------------------------

        for item in entities:

            entity_name = item.get(
                "name"
            )

            entity_type = item.get(
                "entity_type"
            )

            if not entity_name:
                continue

            if entity_type not in VALID_ENTITY_TYPES:
                continue

            if not isinstance(
                entity_name,
                str
            ):
                continue

            entity_name = (
                entity_name
                .strip()
                .lower()
            )

            if not entity_name:
                continue

            # Find existing entity for this user.
            entity = (
                db.query(KnowledgeEntity)
                .filter(
                    KnowledgeEntity.user_id == user_id,
                    KnowledgeEntity.name == entity_name,
                )
                .first()
            )

            # Create entity if it does not exist.
            if entity is None:

                entity = KnowledgeEntity(
                    user_id=user_id,
                    entity_type=entity_type,
                    name=entity_name,
                )

                db.add(entity)
                db.flush()

            # Connect this entity to every emotion/behavior
            # extracted from the same message.
            for memory, tag in extracted_memories:

                edge = KnowledgeEdge(
                    user_id=user_id,
                    entity_id=entity.id,
                    episodic_memory_id=memory.id,
                    tag=tag,
                )

                db.add(edge)

        db.commit()

        # ---------------------------------------------------------
        # Existing pattern synthesis
        # ---------------------------------------------------------

        total_observations = (
            db.query(BehaviorObservation)
            .filter(
                BehaviorObservation.user_id == user_id
            )
            .count()
        )

        if (
            total_observations > 0
            and total_observations % 3 == 0
        ):

            synthesize_patterns_from_observations(
                user_id
            )

    except Exception as e:

        logger.error(
            "Conversation analysis failed: %s",
            e,
            exc_info=True,
        )

        db.rollback()

    finally:

        db.close()

