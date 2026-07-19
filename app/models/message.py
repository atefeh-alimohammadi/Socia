from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base
from sqlalchemy import CheckConstraint


class Message(Base):
    __tablename__ = 'messages'

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="valid_role"),
    )

    id = Column(Integer, primary_key=True)

    session_id = Column(Integer, ForeignKey('conversation_sessions.id'), nullable=False)

    role = Column(String,nullable=False)

    content = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship('ConversationSession', back_populates='messages')
