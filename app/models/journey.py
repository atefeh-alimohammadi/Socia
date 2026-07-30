from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, CheckConstraint, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base


class Journey(Base):
    __tablename__ = 'journeys'

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    title = Column(String(200), nullable=False)

    description = Column(Text, nullable=False)

    source = Column(String(200), nullable=False, server_default="user_created")

    status = Column(String(200), nullable=False, server_default="active")

    day_current = Column(Integer, nullable=False, server_default="1")

    day_total = Column(Integer, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'completed', 'paused')",
            name="journey_status_check"
        ),
        CheckConstraint(
            "source IN ('user_created', 'ai_suggested')",
            name="journey_source_check"
        ),
    )

    user = relationship("User", back_populates="journeys")

    challenges = relationship("JourneyChallenge", back_populates="journey", cascade="all, delete-orphan")
