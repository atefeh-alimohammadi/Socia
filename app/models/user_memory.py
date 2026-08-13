from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    CheckConstraint,
    Float
)

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.database import Base


class UserMemory(Base):

    __tablename__ = "user_memory"


    id = Column(
        Integer,
        primary_key=True
    )


    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )


    memory_type = Column(
        String,
        nullable=False
    )


    content = Column(
        Text,
        nullable=False
    )


    source = Column(
        String,
        nullable=True
    )


    source_type = Column(
        String,
        nullable=True
    )


    evidence_count = Column(
        Integer,
        nullable=False,
        server_default="1"
    )


    confidence = Column(
        Float,
        nullable=True
    )


    version = Column(
        Integer,
        nullable=False,
        server_default="1"
    )


    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )


    tag = Column(
        String,
        nullable=True
    )


    user = relationship(
        "User",
        back_populates="memories"
    )


    history = relationship(
        "UserMemoryHistory",
        back_populates="user_memory",
        cascade="all, delete-orphan"
    )


    evidence_links = relationship(
        "UserMemoryEvidence",
        back_populates="user_memory",
        cascade="all, delete-orphan"
    )


    __table_args__ = (

        CheckConstraint(
            "memory_type IN ('pattern', 'emotion_pattern', 'preference', 'insight')",
            name="check_memory_type"
        ),


        CheckConstraint(
            """
            source_type IN (
                'conversation',
                'onboarding'
            )
            OR source_type IS NULL
            """,
            name="check_memory_source_type"
        ),

    )