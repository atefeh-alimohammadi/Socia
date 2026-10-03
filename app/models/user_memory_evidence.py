from sqlalchemy import (
    Column,
    Integer,
    ForeignKey,
    DateTime,
)

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base


class UserMemoryEvidence(Base):
    __tablename__ = "user_memory_evidence"

    id = Column(
        Integer,
        primary_key=True
    )

    user_memory_id = Column(
        Integer,
        ForeignKey("user_memory.id"),
        nullable=False
    )

    episodic_memory_id = Column(
        Integer,
        ForeignKey("episodic_memory.id", ondelete="CASCADE"),
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    user_memory = relationship(
        "UserMemory",
        back_populates="evidence_links"
    )

    episodic_memory = relationship(
        "EpisodicMemory"
    )