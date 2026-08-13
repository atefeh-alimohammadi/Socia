from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base


class MessageFeedback(Base):
    __tablename__ = "message_feedback"
    id = Column(Integer, primary_key=True)

    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    rating = Column(String, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "rating IN ('helpful', 'not_helpful')",
            name="check_message_feedback_rating",
        ),
        UniqueConstraint(
            "message_id",
            "user_id",
            name="uq_message_feedback_message_user",
        ),
    )

    message = relationship("Message", back_populates="feedback")

    user = relationship("User", back_populates="message_feedbacks")