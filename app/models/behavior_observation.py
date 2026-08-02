from sqlalchemy import (
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


class BehaviorObservation(Base):

    __tablename__ = "behavior_observations"


    id = Column(
        Integer,
        primary_key=True
    )


    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )


    message_id = Column(
        Integer,
        ForeignKey("messages.id"),
        nullable=False
    )


    session_id = Column(
        Integer,
        ForeignKey("conversation_sessions.id"),
        nullable=False
    )


    # نوع رفتار شناسایی شده
    tag = Column(
        String,
        nullable=False
    )


    # بخشی از گفتگو که این رفتار را نشان داده
    evidence = Column(
        Text,
        nullable=False
    )


    # میزان اطمینان AI
    confidence = Column(
        Float,
        nullable=False
    )


    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    user = relationship(
        "User",
        back_populates="behavior_observations"
    )

    message = relationship(
        "Message",
        back_populates="behavior_observations"
    )

    session = relationship(
        "ConversationSession",
        back_populates="behavior_observations"
    )


    __table_args__ = (
        CheckConstraint(
            """
            tag IN (
                'fear_of_judgment',
                'avoiding_social_expression',
                'overthinking',
                'low_confidence',
                'social_avoidance',
                'seeking_validation',
                'conflict_avoidance'
            )
            """,
            name="check_behavior_observation_tag"
        ),
    )