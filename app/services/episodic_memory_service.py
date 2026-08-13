import logging

from sqlalchemy.orm import Session
from ollama import embeddings as ollama_embeddings

from app.models.episodic_memory import EpisodicMemory


logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"


def generate_embedding(
    text: str,
) -> list[float] | None:

    try:

        response = ollama_embeddings(
            model=EMBEDDING_MODEL,
            prompt=text,
        )

        return response["embedding"]

    except Exception as e:

        logger.error(
            "Embedding generation failed: %s",
            e,
            exc_info=True,
        )

        return None


def create_episodic_memory(
    db: Session,
    user_id: int,
    session_id: int,
    message_id: int,
    event_type: str,
    tag: str,
    content: str,
    confidence: float,
    importance: float | None = None,
) -> EpisodicMemory:

    memory = EpisodicMemory(
        user_id=user_id,
        session_id=session_id,
        message_id=message_id,
        event_type=event_type,
        tag=tag,
        content=content,
        confidence=confidence,
        importance=(
            importance
            if importance is not None
            else confidence
        ),
        embedding=generate_embedding(content),
    )

    db.add(memory)

    return memory


def get_episodic_memories_for_user(
    db: Session,
    user_id: int,
    event_type: str | None = None,
    limit: int | None = None,
) -> list[EpisodicMemory]:

    query = (
        db.query(EpisodicMemory)
        .filter(
            EpisodicMemory.user_id == user_id
        )
        .order_by(
            EpisodicMemory.created_at.desc()
        )
    )

    if event_type:

        query = query.filter(
            EpisodicMemory.event_type == event_type
        )

    if limit:

        query = query.limit(limit)

    return query.all()

def retrieve_relevant_episodic_memories(
    db: Session,
    user_id: int,
    query_text: str,
    limit: int = 5,
    min_importance: float = 0.0,
    max_distance: float = 0.45,
) -> list[EpisodicMemory]:

    query_embedding = generate_embedding(query_text)

    if query_embedding is None:
        logger.warning(
            "Query embedding generation failed; "
            "falling back to recency-based retrieval for user_id=%s",
            user_id,
        )
        return get_episodic_memories_for_user(
            db,
            user_id,
            limit=limit,
        )

    distance = EpisodicMemory.embedding.cosine_distance(query_embedding)

    # For each source message, keep only its single closest-matching
    # episodic memory, so one message (which can produce multiple emotion/
    # behavior rows) cannot dominate the result set.
    best_per_message = (
        db.query(
            EpisodicMemory.id.label("id"),
            distance.label("distance"),
        )
        .filter(
            EpisodicMemory.user_id == user_id,
            EpisodicMemory.embedding.is_not(None),
            EpisodicMemory.importance >= min_importance,
            distance <= max_distance,
        )
        .order_by(
            EpisodicMemory.message_id,
            distance,
        )
        .distinct(EpisodicMemory.message_id)
        .subquery()
    )

    results = (
        db.query(EpisodicMemory)
        .join(
            best_per_message,
            EpisodicMemory.id == best_per_message.c.id,
        )
        .order_by(best_per_message.c.distance)
        .limit(limit)
        .all()
    )

    if not results:
        logger.info(
            "No episodic memories within distance threshold for user_id=%s (query=%r)",
            user_id,
            query_text,
        )

    return results
