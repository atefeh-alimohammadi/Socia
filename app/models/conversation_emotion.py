from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    CheckConstraint,
)

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base


class ConversationEmotion(Base):

    __tablename__ = "conversation_emotions"

    id = Column(
        Integer,
        primary_key=True
    )

    message_id = Column(
        Integer,
        ForeignKey("messages.id"),
        nullable=False
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )

    emotion = Column(
        String,
        nullable=False
    )

    intensity = Column(
        Float,
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


    user = relationship(
        "User",
        back_populates="conversation_emotions"
    )


    message = relationship(
        "Message",
        back_populates="conversation_emotions"
    )

    user = relationship(
        "User",
        back_populates="conversation_emotions"
    )

    message = relationship(
        "Message",
        back_populates="conversation_emotions"
    )


    __table_args__ = (
        CheckConstraint(
            """
            emotion IN (
                'anxiety',
                'confidence',
                'fear',
                'excitement',
                'frustration',
                'sadness',
                'shame',
                'hope'
            )
            """,
            name="check_conversation_emotion"
        ),
    )