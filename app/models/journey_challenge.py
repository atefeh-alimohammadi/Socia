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

    skip_reason = Column(String, nullable=True)

    skip_reason_detail = Column(Text, nullable=True)

    difficulty_feedback = Column(String, nullable=True)

    emotional_response = Column(String, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'completed', 'skipped')",
            name="journey_challenge_status_check"
        ),
        CheckConstraint(
            """
            skip_reason IN (
                'busy',
                'forgot',
                'too_difficult',
                'too_anxious',
                'not_relevant',
                'disliked_activity',
                'situation_unavailable',
                'other'
            )
            OR skip_reason IS NULL
            """,
            name="journey_challenge_skip_reason_check"
        ),
        CheckConstraint(
            """
            difficulty_feedback IN ('too_easy', 'just_right', 'too_hard')
            OR difficulty_feedback IS NULL
            """,
            name="journey_challenge_difficulty_feedback_check"
        ),
    )

    journey = relationship("Journey", back_populates="challenges")