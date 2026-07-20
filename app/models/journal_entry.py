from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base

class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True)

    title = Column(String, nullable=False)

    content = Column(Text, nullable=False)

    mood = Column(String, nullable=True)

    analysis_status = Column(String, server_default="pending")

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    owner = relationship("User", back_populates="journal_entries")

    emotion_analysis = relationship("EmotionAnalysis", back_populates="journal_entry", cascade="all, delete-orphan")
