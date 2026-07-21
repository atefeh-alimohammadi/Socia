from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session

from app.database.deps import get_db
from app.api.deps import get_current_user

from app.models.user import User
from app.models.user_memory import UserMemory

from app.schemas.user_memory import UserMemoryResponse
from app.services.memory_service import synthesize_user_patterns

router = APIRouter(
    prefix="/memory",
    tags=["Memory"]
)

@router.get(
    "/",
    response_model=list[UserMemoryResponse]
  )
def get_memories(
    db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
  ):
    memories = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == current_user.id)
        .all()
    )
    return memories

@router.post(
    "/synthesize",
    status_code=status.HTTP_202_ACCEPTED,

)
def synthesize_memories(
        background_tasks: BackgroundTasks,
        current_user: User = Depends(get_current_user),
):
    background_tasks.add_task(synthesize_user_patterns, current_user.id)

    return {
        "message": "Pattern synthesis started"
    }

