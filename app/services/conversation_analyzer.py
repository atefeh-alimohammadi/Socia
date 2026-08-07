import json
import logging

from app.ai.ollama import generate_ai_response
from app.database.database import SessionLocal

from app.models.conversation_emotion import ConversationEmotion
from app.models.behavior_observation import BehaviorObservation

from app.services.memory_service import synthesize_patterns_from_observations


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


def build_analyzer_prompt(content: str):

    return f"""
You are analyzing a user's conversation message for social behavior signals.

User message:

"{content}"


Extract emotions and behavior observations.

Return ONLY valid JSON:

{{
 "emotions":[
    {{
      "emotion":"anxiety",
      "intensity":0.8
    }}
 ],
 "observations":[
    {{
      "tag":"fear_of_judgment",
      "evidence":"User expressed fear of being judged by others.",
      "confidence":0.85
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


Rules:
- Only extract clearly present signals.
- Do not invent patterns.
- Intensity and confidence between 0 and 1.
- Evidence must be one sentence.
- Return empty lists if nothing exists.
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


        for item in emotions:

            emotion_name = item.get(
                "emotion"
            )

            intensity = item.get(
                "intensity"
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


            emotion = ConversationEmotion(
                message_id=message_id,
                user_id=user_id,
                emotion=emotion_name,
                intensity=intensity,
            )

            db.add(emotion)



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



        db.commit()



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
            exc_info=True
        )

        db.rollback()



    finally:

        db.close()