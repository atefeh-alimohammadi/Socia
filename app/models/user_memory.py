from sqlalchemy import (
Column,
Integer,
String,
Text,
DateTime,
ForeignKey,
CheckConstraint
)

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base


class UserMemory(Base):
    __tablename__ = "user_memory"

    id = Column(Integer, primary_key=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )

    memory_type = Column(String, nullable=False)

    content = Column(Text, nullable=False)

    source = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="memories")

    __table_args__ = (
        CheckConstraint(
            "memory_type IN ('pattern', 'preference', 'insight')",
            name="check_memory_type"
        ),
    )