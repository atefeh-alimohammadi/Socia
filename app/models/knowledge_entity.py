from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String,
    DateTime,
    CheckConstraint,
    UniqueConstraint,
)

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base


class KnowledgeEntity(Base):
    __tablename__ = "knowledge_entities"

    id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    entity_type = Column(String, nullable=False)

    name = Column(String(200), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "entity_type IN ('person', 'situation', 'topic')",
            name="knowledge_entity_check",
        ),
        UniqueConstraint(
            "user_id",
            "name",
            name="knowledge_entity_user_name",
        ),
    )

    user = relationship("User", back_populates="knowledge_entities")

    edges = relationship("KnowledgeEdge", back_populates="entity", cascade="all, delete-orphan")