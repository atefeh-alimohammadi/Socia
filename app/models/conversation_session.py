from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    title = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    owner = relationship('User', back_populates='conversation_sessions')

    messages = relationship('Message', back_populates='session', cascade='all, delete-orphan')

    behavior_observations = relationship(
        "BehaviorObservation",
        back_populates="session",
        cascade="all, delete-orphan"
    )


