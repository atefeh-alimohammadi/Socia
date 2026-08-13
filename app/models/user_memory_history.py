from sqlalchemy import (
    Column,
    Integer,
    Text,
    Float,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base


class UserMemoryHistory(Base):
    __tablename__ = "user_memory_history"

    id = Column(
        Integer,
        primary_key=True
    )

    user_memory_id = Column(
        Integer,
        ForeignKey("user_memory.id"),
        nullable=False
    )

    # Snapshot of the previous state of UserMemory
    content = Column(
        Text,
        nullable=False
    )

    confidence = Column(
        Float,
        nullable=True
    )

    evidence_count = Column(
        Integer,
        nullable=False,
        default=0
    )

    version = Column(
        Integer,
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    user_memory = relationship(
        "UserMemory",
        back_populates="history"
    )