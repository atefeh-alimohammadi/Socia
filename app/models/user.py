from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from app.database.database import Base
from sqlalchemy.orm import relationship
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String,nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, server_default="true")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    onboarding_completed = Column(Boolean, server_default="false")

    journal_entries = relationship("JournalEntry", back_populates="owner", cascade="all, delete-orphan")

    conversation_sessions = relationship("ConversationSession", back_populates="owner", cascade="all, delete-orphan")

    memories = relationship("UserMemory", back_populates="user", cascade="all, delete-orphan")

    journeys = relationship(
        "Journey",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    conversation_emotions = relationship(
        "ConversationEmotion",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    behavior_observations = relationship(
        "BehaviorObservation",
        back_populates="user",
        cascade="all, delete-orphan"
    )

