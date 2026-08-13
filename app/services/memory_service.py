import logging

from sqlalchemy import func
from ollama import chat

from app.database.database import SessionLocal

from app.models.user_memory import UserMemory
from app.models.episodic_memory import EpisodicMemory
from app.models.user_memory_history import UserMemoryHistory
from app.models.user_memory_evidence import UserMemoryEvidence


CONFIDENCE_CHANGE_THRESHOLD = 0.05

logger = logging.getLogger(__name__)


PATTERN_GENERATION_PROMPT = """
You are generating a memory summary for a user.

Memory category:
{event_type}

Tag:
{tag}

Evidence from conversations:

{evidence}

Write a single sentence describing this recurring pattern.

Write in third person.

Be specific, supportive, and avoid clinical or diagnostic language.
Describe observations, not personality flaws.

Return only the sentence.
"""


def _snapshot_current_state(
    db,
    memory: UserMemory
) -> None:

    db.add(
        UserMemoryHistory(
            user_memory_id=memory.id,
            content=memory.content,
            confidence=memory.confidence,
            evidence_count=memory.evidence_count,
            version=memory.version,
        )
    )


def _link_new_evidence(
    db,
    user_memory_id: int,
    episodic_ids: list[int]
) -> None:

    already_linked = {
        row.episodic_memory_id
        for row in (
            db.query(
                UserMemoryEvidence.episodic_memory_id
            )
            .filter(
                UserMemoryEvidence.user_memory_id == user_memory_id
            )
            .all()
        )
    }


    for episodic_id in episodic_ids:

        if episodic_id in already_linked:
            continue


        db.add(
            UserMemoryEvidence(
                user_memory_id=user_memory_id,
                episodic_memory_id=episodic_id,
            )
        )


def synthesize_patterns_from_observations(
    user_id: int,
) -> None:


    db = SessionLocal()


    try:


        tag_counts = (
            db.query(
                EpisodicMemory.event_type,
                EpisodicMemory.tag,
                func.count(
                    EpisodicMemory.id
                ).label("count"),
                func.avg(
                    EpisodicMemory.confidence
                ).label("avg_confidence")
            )
            .filter(
                EpisodicMemory.user_id == user_id,
                EpisodicMemory.event_type.in_(
                    [
                        "behavior",
                        "emotion"
                    ]
                )
            )
            .group_by(
                EpisodicMemory.event_type,
                EpisodicMemory.tag
            )
            .all()
        )


        for event_type, tag, count, avg_confidence in tag_counts:


            if count < 3:
                continue


            new_confidence = round(
                avg_confidence,
                2
            )


            memory_type = (
                "emotion_pattern"
                if event_type == "emotion"
                else "pattern"
            )


            evidence_rows = (
                db.query(EpisodicMemory)
                .filter(
                    EpisodicMemory.user_id == user_id,
                    EpisodicMemory.tag == tag,
                    EpisodicMemory.event_type == event_type
                )
                .order_by(
                    EpisodicMemory.created_at.desc()
                )
                .limit(5)
                .all()
            )


            evidence_ids = [
                row.id
                for row in evidence_rows
            ]


            existing = (
                db.query(UserMemory)
                .filter(
                    UserMemory.user_id == user_id,
                    UserMemory.tag == tag,
                    UserMemory.source_type == "conversation",
                    UserMemory.memory_type == memory_type
                )
                .first()
            )


            if existing:


                unchanged = (
                    existing.evidence_count == count
                    and existing.confidence is not None
                    and abs(
                        existing.confidence - new_confidence
                    ) < CONFIDENCE_CHANGE_THRESHOLD
                )


                if unchanged:

                    _link_new_evidence(
                        db,
                        existing.id,
                        evidence_ids
                    )

                    continue



                _snapshot_current_state(
                    db,
                    existing
                )


                existing.evidence_count = count
                existing.confidence = new_confidence
                existing.version += 1


                _link_new_evidence(
                    db,
                    existing.id,
                    evidence_ids
                )


                continue



            evidence_text = "\n".join(
                [
                    row.content
                    for row in evidence_rows
                ]
            )


            prompt = PATTERN_GENERATION_PROMPT.format(
                event_type=event_type,
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
                memory_type=memory_type,
                tag=tag,
                content=generated_description,
                source="conversation",
                source_type="conversation",
                evidence_count=count,
                confidence=new_confidence,
                version=1,
            )


            db.add(memory)

            db.flush()


            _link_new_evidence(
                db,
                memory.id,
                evidence_ids
            )


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