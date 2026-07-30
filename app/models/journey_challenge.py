from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base


class JourneyChallenge(Base):
    __tablename__ = "journey_challenges"

    id = Column(Integer, primary_key=True, index=True)

    journey_id = Column(Integer, ForeignKey("journeys.id", ondelete="CASCADE"), nullable=False)

    day_number = Column(Integer, nullable=False)

    title = Column(String(200), nullable=False)

    description = Column(Text, nullable=False)

    status = Column(String, nullable=False, default="pending")

    completed_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'completed', 'skipped')",
            name="journey_challenge_status_check"
        ),
    )

    journey = relationship("Journey", back_populates="challenges")
