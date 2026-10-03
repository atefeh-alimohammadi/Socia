from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base
from sqlalchemy import CheckConstraint


class Message(Base):
    __tablename__ = 'messages'

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="valid_role"),
    )

    id = Column(Integer, primary_key=True)

    session_id = Column(Integer, ForeignKey('conversation_sessions.id'), nullable=False)

    role = Column(String,nullable=False)

    content = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship('ConversationSession', back_populates='messages')

    conversation_emotions = relationship(
        "ConversationEmotion",
        back_populates="message",
        cascade="all, delete-orphan"
    )

    behavior_observations = relationship(
        "BehaviorObservation",
        back_populates="message",
        cascade="all, delete-orphan"
    )

    episodic_memories = relationship(
        "EpisodicMemory",
        back_populates="message",
        cascade="all, delete-orphan"
    )

    feedback = relationship(
        "MessageFeedback",
        back_populates="message",
        uselist=False,
        cascade="all, delete-orphan"
    )

