from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, TEXT
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base


class EmotionAnalysis(Base):
    __tablename__ = "emotion_analysis"
    id = Column(Integer, primary_key=True)

    journal_entry_id = Column(Integer, ForeignKey('journal_entries.id'), nullable=False)

    emotion = Column(String, nullable=False)

    confidence_score = Column(Float, nullable=False)

    notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(),)

    journal_entry = relationship("JournalEntry", back_populates="emotion_analysis")
