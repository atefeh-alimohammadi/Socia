from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.conversation_session import ConversationSession


def get_user_conversation(
        db: Session,
        session_id: int,
        user_id: int,
) -> ConversationSession:

    conversation = (
        db.query(ConversationSession)
        .filter(ConversationSession.id == session_id)
        .first()

    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )

    if conversation.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized to view this conversation",
        )

    return conversation
