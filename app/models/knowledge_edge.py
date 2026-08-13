from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base


class KnowledgeEdge(Base):
    __tablename__ = "knowledge_edges"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    entity_id = Column(
        Integer,
        ForeignKey("knowledge_entities.id", ondelete="CASCADE"),
        nullable=False,
    )

    episodic_memory_id = Column(
        Integer,
        ForeignKey("episodic_memory.id", ondelete="CASCADE"),
        nullable=False,
    )

    tag = Column(
        String(100),
        nullable=False,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="knowledge_edges",
    )

    entity = relationship(
        "KnowledgeEntity",
        back_populates="edges",
    )

    episodic_memory = relationship(
        "EpisodicMemory",
        back_populates="knowledge_edges",
    )