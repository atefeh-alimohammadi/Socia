
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.knowledge_entity import KnowledgeEntity
from app.models.knowledge_edge import KnowledgeEdge


def get_related_entities_for_tag(
    db: Session,
    user_id: int,
    tag: str,
):
    """
    Return entities most frequently associated with a given tag
    for a specific user.

    Results are ordered by frequency, highest first.
    """

    results = (
        db.query(
            KnowledgeEntity.id,
            KnowledgeEntity.name,
            KnowledgeEntity.entity_type,
            func.count(KnowledgeEdge.id).label("frequency"),
        )
        .join(
            KnowledgeEdge,
            KnowledgeEdge.entity_id == KnowledgeEntity.id,
        )
        .filter(
            KnowledgeEntity.user_id == user_id,
            KnowledgeEdge.user_id == user_id,
            KnowledgeEdge.tag == tag,
        )
        .group_by(
            KnowledgeEntity.id,
            KnowledgeEntity.name,
            KnowledgeEntity.entity_type,
        )
        .order_by(
            func.count(KnowledgeEdge.id).desc()
        )
        .all()
    )

    return [
        {
            "entity_id": entity_id,
            "name": name,
            "entity_type": entity_type,
            "frequency": frequency,
        }
        for (
            entity_id,
            name,
            entity_type,
            frequency,
        ) in results
    ]

