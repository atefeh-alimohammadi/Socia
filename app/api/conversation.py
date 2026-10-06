from fastapi import APIRouter, Depends, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.schemas.conversation import (
    SessionCreate,
    SessionResponse,
    MessageCreate,
    MessageResponse,
)

from app.services.conversation_title_service import (
    generate_conversation_title,
)

from app.models.user import User
from app.models.conversation_session import ConversationSession
from app.models.message import Message
from app.models.user_memory import UserMemory

from app.api.deps import get_current_user
from app.database.deps import get_db

from app.services.authorization import get_user_conversation
from app.services.conversation_analyzer import analyze_conversation_message
from app.services.episodic_memory_service import (
    retrieve_relevant_episodic_memories
)

from app.ai.ollama import get_ai_response
from app.services.context_assembler import assemble_user_context
from app.services.safety_service import check_message_safety
from app.services.orchestrator import route_message
from app.services.reflection_service import check_response_consistency
from app.services.personalization_service import get_communication_policy

import logging


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/conversation",
    tags=["Conversation"]
)


@router.post(
    "/",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED
)
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


@router.post(
    "/{session_id}/messages",
    response_model=list[MessageResponse],
    status_code=status.HTTP_201_CREATED
)
def send_message(
    session_id: int,
    message: MessageCreate,
    background_tasks: BackgroundTasks,
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

    safety_result = check_message_safety(user_message.content)

    if safety_result["risk_level"] == "high":
        safety_response = (
            "I'm really sorry you're going through this. "
            "You don't have to handle this moment alone. "
            "If you might act on these thoughts or you're in immediate danger, "
            "please contact your local emergency service or go to the nearest "
            "emergency department. If you can, stay with someone you trust "
            "and move away from anything you could use to hurt yourself."
        )

        assistant_message = Message(
            session_id=conversation.id,
            role="assistant",
            content=safety_response,
        )

        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)

        return [
            user_message,
            assistant_message,
        ]

    route = route_message(
        user_message.content
    )

    conversation_history = (
        db.query(Message)
        .filter(Message.session_id == conversation.id)
        .order_by(Message.created_at)
        .limit(20)
        .all()
    )

    conversation_history.reverse()

    user_memories = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == current_user.id)
        .all()
    )

    if route["needs_episodic_retrieval"]:
        relevant_episodic_memories = (
            retrieve_relevant_episodic_memories(
                db,
                user_id=current_user.id,
                query_text=user_message.content,
            )
        )
    else:
        relevant_episodic_memories = []

    if route["needs_full_context"]:
        user_context = assemble_user_context(
            current_user.id,
            db,
        )
    else:
        user_context = {
            "recent_emotions": [],
            "active_journeys": [],
        }

    # Milestone 12:
    # Compute the user's communication policy from existing feedback.
    # This is deterministic and does not require an LLM call.
    communication_policy = get_communication_policy(
        db=db,
        user_id=current_user.id,
    )

    logger.info(
        "Communication policy for user %s: %s",
        current_user.id,
        communication_policy,
    )

    ai_response = get_ai_response(
        user_message.content,
        conversation_history=conversation_history,
        user_memories=user_memories,
        relevant_episodic_memories=relevant_episodic_memories,
        recent_emotions=user_context["recent_emotions"],
        active_journeys=user_context["active_journeys"],
        communication_policy=communication_policy["pacing"],
    )

    reflection_result = check_response_consistency(
        response=ai_response,
        user_memories=user_memories,
        recent_emotions=user_context["recent_emotions"],
    )

    if not reflection_result["consistent"]:
        logger.warning(
            "Reflection detected inconsistency: %s",
            reflection_result["issue"],
        )

        correction_prompt = f"""
The previous response may be inconsistent with the user's context.

Issue detected:
{reflection_result["issue"]}

Generate a corrected response to the user's original message.

Do not mention this reflection process to the user.
"""

        ai_response = get_ai_response(
            user_message.content,
            conversation_history=conversation_history,
            user_memories=user_memories,
            relevant_episodic_memories=relevant_episodic_memories,
            recent_emotions=user_context["recent_emotions"],
            active_journeys=user_context["active_journeys"],
            communication_policy=communication_policy["pacing"],
            additional_context=correction_prompt,
        )

        retry_reflection = check_response_consistency(
            response=ai_response,
            user_memories=user_memories,
            recent_emotions=user_context["recent_emotions"],
        )

        if retry_reflection["consistent"]:
            logger.info(
                "Reflection outcome: corrected"
            )
        else:
            logger.warning(
                "Reflection outcome: uncorrected_after_retry issue=%s",
                retry_reflection["issue"],
            )

    else:
        logger.info(
            "Reflection outcome: pass"
        )


    assistant_message = Message(
        session_id=conversation.id,
        role="assistant",
        content=ai_response,
    )

    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)

    if conversation.title is None:
        user_message_count = (
            db.query(Message)
            .filter(
                Message.session_id == conversation.id,
                Message.role == "user",
            )
            .count()
        )

        if user_message_count == 1:
            generated_title = generate_conversation_title(
                user_message=user_message.content,
                assistant_response=assistant_message.content,
            )

            if generated_title:
                conversation.title = generated_title

                db.add(conversation)
                db.commit()
                db.refresh(conversation)


    background_tasks.add_task(
        analyze_conversation_message,
        message_id=user_message.id,
        user_id=current_user.id,
        session_id=session_id,
        content=user_message.content,
    )

    return [
        user_message,
        assistant_message,
    ]


@router.get(
    "/",
    response_model=list[SessionResponse],
    status_code=status.HTTP_200_OK
)
def get_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conversations = (
        db.query(ConversationSession)
        .filter(
            ConversationSession.user_id == current_user.id
        )
        .order_by(
            ConversationSession.created_at.desc()
        )
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

    db.delete(conversation)
    db.commit()