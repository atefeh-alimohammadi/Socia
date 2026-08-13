from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.deps import get_db
from app.api.deps import get_current_user

from app.models.user import User
from app.models.message import Message
from app.models.message_feedback import MessageFeedback

from app.services.authorization import get_user_conversation


router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"],
)


class FeedbackCreate(BaseModel):
    rating: str


@router.post(
    "/conversation/{session_id}/messages/{message_id}/feedback",
    status_code=status.HTTP_200_OK,
)
def submit_message_feedback(
    session_id: int,
    message_id: int,
    feedback: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if feedback.rating not in {
        "helpful",
        "not_helpful",
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rating must be 'helpful' or 'not_helpful'",
        )

    conversation = get_user_conversation(
        db=db,
        session_id=session_id,
        user_id=current_user.id,
    )

    message = (
        db.query(Message)
        .filter(
            Message.id == message_id,
            Message.session_id == conversation.id,
            Message.role == "assistant",
        )
        .first()
    )

    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )

    existing_feedback = (
        db.query(MessageFeedback)
        .filter(
            MessageFeedback.message_id == message.id,
            MessageFeedback.user_id == current_user.id,
        )
        .first()
    )

    if existing_feedback:
        existing_feedback.rating = feedback.rating
    else:
        new_feedback = MessageFeedback(
            message_id=message.id,
            user_id=current_user.id,
            rating=feedback.rating,
        )

        db.add(new_feedback)

    db.commit()

    return {
        "message_id": message.id,
        "rating": feedback.rating,
    }