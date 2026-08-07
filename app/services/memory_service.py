import json
import logging
from sqlalchemy import func

from ollama import chat

from app.database.database import SessionLocal

from app.models.behavior_observation import BehaviorObservation
from app.models.user_memory import UserMemory


logger = logging.getLogger(__name__)

PATTERN_GENERATION_PROMPT = """
You are generating a behavioral pattern summary for a user.

Behavioral tag:
{tag}

Evidence from conversations:

{evidence}

Write a single sentence describing this recurring pattern.

Write in third person.

Be specific, supportive, and avoid clinical or diagnostic language.
Describe behaviors, not personality flaws.

Return only the sentence.
"""

def synthesize_patterns_from_observations(
        user_id: int,
) -> None:

    db = SessionLocal()

    try:

        tag_counts = (
            db.query(
                BehaviorObservation.tag,
                func.count(
                    BehaviorObservation.id
                ).label("count"),
                func.avg(
                    BehaviorObservation.confidence
                ).label("avg_confidence")
            )
            .filter(
                BehaviorObservation.user_id == user_id
            )
            .group_by(
                BehaviorObservation.tag
            )
            .all()
        )


        for tag, count, avg_confidence in tag_counts:

            if count < 3:
                continue


            existing = (
                db.query(UserMemory)
                .filter(
                    UserMemory.user_id == user_id,
                    UserMemory.tag == tag,
                    UserMemory.source_type == "conversation"
                )
                .first()
            )


            if existing:

                existing.evidence_count = count

                existing.confidence = round(
                    avg_confidence,
                    2
                )

                db.commit()

                continue



            evidence_rows = (
                db.query(BehaviorObservation)
                .filter(
                    BehaviorObservation.user_id == user_id,
                    BehaviorObservation.tag == tag
                )
                .limit(5)
                .all()
            )


            evidence_text = "\n".join(
                [
                    row.evidence
                    for row in evidence_rows
                ]
            )


            prompt = PATTERN_GENERATION_PROMPT.format(
                tag=tag,
                evidence=evidence_text
            )


            response = chat(
                model="qwen2.5:7b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )


            generated_description = (
                response.message.content.strip()
            )


            memory = UserMemory(
                user_id=user_id,
                memory_type="pattern",
                tag=tag,
                content=generated_description,
                source="conversation",
                source_type="conversation",
                evidence_count=count,
                confidence=round(
                    avg_confidence,
                    2
                )
            )


            db.add(memory)


        db.commit()


    except Exception as e:

        logger.error(
            "Pattern synthesis failed: %s",
            e,
            exc_info=True
        )

        db.rollback()


    finally:

        db.close()