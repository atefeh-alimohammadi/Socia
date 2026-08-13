from sqlalchemy import(
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    CheckConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base

from pgvector.sqlalchemy import Vector



class EpisodicMemory(Base):
    __tablename__ = "episodic_memory"

    id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    session_id = Column(
        Integer,
        ForeignKey('conversation_sessions.id'),
        nullable=False,
    )

    message_id = Column(
        Integer,
        ForeignKey("messages.id"),
        nullable=False,
    )

    event_type = Column(String, nullable=False)

    tag = Column(String, nullable=False)

    content = Column(Text, nullable=False)

    confidence = Column(Float, nullable=False)

    importance = Column(Float, nullable=False)

    embedding = Column(
        Vector(768),
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    user = relationship("User", back_populates="episodic_memories")

    session = relationship("ConversationSession")

    message = relationship(
        "Message",
        back_populates="episodic_memories"
    )

    knowledge_edges = relationship(
        "KnowledgeEdge",
        back_populates="episodic_memory",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "event_type IN ('emotion', 'behavior')",
            name="check_episodic_memory_event_type",
        ),

        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="check_episodic_memory_confidence",
        ),
        CheckConstraint(
            "importance >= 0 AND importance <= 1",
            name="check_episodic_memory_importance",
        ),
    )