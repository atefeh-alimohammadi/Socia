from fastapi import APIRouter, Depends, BackgroundTasks, status
from sqlalchemy.orm import Session

from app.database.deps import get_db
from app.api.deps import get_current_user

from app.models.user import User
from app.models.user_memory import UserMemory

from app.schemas.user_memory import UserMemoryResponse
from app.services.memory_service import synthesize_patterns_from_observations

router = APIRouter(
    prefix="/memory",
    tags=["Memory"]
)

@router.get(
    "/",
    response_model=list[UserMemoryResponse]
  )
def get_memories(
        memory_type: str | None = None,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
  ):
    query = (
        db.query(UserMemory)
        .filter(
            UserMemory.user_id == current_user.id
        )
    )

    if memory_type:
        query = query.filter(
            UserMemory.memory_type == memory_type
        )

    memories = query.order_by(
        UserMemory.created_at.desc()
    ).all()

    return memories

@router.post(
    "/synthesize",
    status_code=status.HTTP_202_ACCEPTED,

)
def synthesize_memories(
        background_tasks: BackgroundTasks,
        current_user: User = Depends(get_current_user),
):
    background_tasks.add_task(synthesize_patterns_from_observations, current_user.id)

    return {
        "message": "Pattern synthesis started"
    }

