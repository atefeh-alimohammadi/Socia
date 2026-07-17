from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.params import Depends
from starlette import status
from sqlalchemy.orm import Session
from app.schemas.conversation import (
SessionCreate,
SessionResponse,
MessageCreate,
MessageResponse,
)

from app.models.user import User
from app.models.conversation_session import ConversationSession
from app.models.message import Message
from app.api.deps import get_current_user
from app.database.deps import get_db
from app.ai.stub import get_ai_response

from app.services.authorization import get_user_conversation
router = APIRouter(prefix="/conversation", tags=["Conversation"])


@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    session_data: SessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    new_session = ConversationSession(
        title=session_data.title,
        user_id=current_user.id,
    )

    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    return new_session

@router.post("/{session_id}/messages", response_model=list[MessageResponse], status_code=status.HTTP_201_CREATED)
def send_message(
        session_id: int,
        message: MessageCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):
    conversation = get_user_conversation(
        db=db,
        session_id=session_id,
        user_id=current_user.id
    )
    user_message = Message(
        session_id=conversation.id,
        role="user",
        content=message.content
    )

    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    conversation_history = (
        db.query(Message)
        .filter(Message.session_id == conversation.id)
        .order_by(Message.created_at)
        .all()
    )

    ai_response = get_ai_response(
        user_message.content,
        conversation_history=conversation_history,
    )

    assistant_message = Message(
        session_id=conversation.id,
        role="assistant",
        content=ai_response
    )

    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)

    return [
        user_message,
        assistant_message,
    ]

@router.get("/", response_model=list[SessionResponse], status_code=status.HTTP_200_OK)
def get_conversations(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):

    conversations = (db.query(ConversationSession)
                     .filter(ConversationSession.user_id == current_user.id)
                     .order_by(ConversationSession.created_at.desc())
                     .all()
                     )
    return conversations

@router.get(
    "/{session_id}",
    response_model=SessionResponse,
)
def get_conversation(
        session_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):

    # conversation = (
    #     db.query(ConversationSession)
    #     .filter(ConversationSession.id == session_id)
    #     .first()
    # )
    #
    # if conversation is None:
    #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    #
    # if conversation.user_id != current_user.id:
    #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not allowed to use this endpoint")
    conversation = get_user_conversation(
        db=db,
        session_id=session_id,
        user_id=current_user.id
    )

    return conversation

@router.get(
    "/{session_id}/messages",
    response_model=list[MessageResponse]
)
def get_messages(
        session_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    conversation = get_user_conversation(
        db=db,
        session_id=session_id,
        user_id=current_user.id
    )

    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not allowed to use this endpoint")

    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    return messages

@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
def delete_conversation(
        session_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    conversation = get_user_conversation(
        db=db,
        session_id=session_id,
        user_id=current_user.id
    )

    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not allowed to use this endpoint")

    db.delete(conversation)
    db.commit()
