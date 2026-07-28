from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.deps import get_db
from app.models.user import User
from app.models.user_memory import UserMemory

from app.api.deps import get_current_user

from ollama import chat
from app.core.onboarding import ONBOARDING_STEPS

from app.schemas.onboarding import OnboardingAnswerCreate, OnboardingCompleteResponse


router = APIRouter(
    prefix="/onboarding",
    tags=["Onboarding"]
)


@router.get("/steps")
def get_steps():

    return ONBOARDING_STEPS


@router.post("/answer")
def submit_onboarding_answer(
        data: OnboardingAnswerCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):

    step_data = next(
        (
            step for step in ONBOARDING_STEPS
            if step["step"] == data.step
        ),
        None
    )

    if not step_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Invalid onboarding Step")

    existing_memory = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == current_user.id,
                UserMemory.tag == step_data["tag"])
        .first()
    )

    if existing_memory:

        existing_memory.content = data.answer

        db.commit()
        db.refresh(existing_memory)

        return existing_memory

    new_memory = UserMemory(
        user_id=current_user.id,
        memory_type=step_data["memory_type"],
        tag=step_data["tag"],
        content=data.answer,
        source="onboarding"
    )

    db.add(new_memory)
    db.commit()
    db.refresh(new_memory)

    return new_memory


@router.post("/complete", response_model=OnboardingCompleteResponse)
def complete_onboarding_answer(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):

    tags = [
        "preferred_name",
        "primary_motivation",
        "communication_challenge",
        "emotional_response_pattern",
        "growth_goal",
    ]

    memories = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == current_user.id,
                UserMemory.tag.in_(tags))
        .all()

    )

    if len(memories)< 5:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Please complete all onboarding steps first")

    user_context = "\n".join([
        f"{memory.tag}: {memory.content}"
        for memory in memories
    ])

    prompt = f"""
    You are Socia, an AI companion.

    The user just completed onboarding.
    Write a warm, personal welcome — exactly 3 sentences.

    Rules:
    - Speak directly to the user using their preferred name.
    - Reference their specific goals and challenges naturally.
    - End with an encouraging statement about what's ahead.
    - Do NOT ask questions.
    - Do NOT give advice yet.
    - Do NOT exceed 3 sentences.

    User answers:

    {user_context}
    """

    try:
        response = chat(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        summary = response.message.content
    except Exception:
        summary = (
            "Welcome to Socia. "
            "I learned a little about your goals and what you want to improve. "
            "I'm here to support your reflection and personal growth."
        )

    current_user.onboarding_completed = True

    db.commit()

    return {
        "summary": summary,
        "onboarding_completed": True,
    }