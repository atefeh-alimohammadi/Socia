from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.database.deps import get_db
from app.models.journal_entry import JournalEntry
from app.models.user import User

from app.schemas.journal_entry import (
JournalEntryCreate,
JournalEntryUpdate,
JournalEntryResponse
)


from app.api.deps import get_current_user
from app.services.journal_service import get_user_journal
from app.services.emotion_analysis import analyze_emotions

from app.models.emotion_analysis import EmotionAnalysis

from app.schemas.emotion_analysis import EmotionAnalysisResponse

router = APIRouter(
    prefix="/journal",
    tags=["journal"],
)

@router.post("/", response_model=JournalEntryResponse, status_code=status.HTTP_201_CREATED)
def create_journal_entry(
        entry: JournalEntryCreate,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    db_entry = JournalEntry(
        title=entry.title,
        content=entry.content,
        mood=entry.mood,
        user_id=current_user.id
    )

    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)

    background_tasks.add_task(analyze_emotions, db_entry.id, db_entry.content)


    return db_entry

@router.get("/", response_model=list[JournalEntryResponse])
def get_my_journals(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)

):
    entries = db.query(JournalEntry).filter(JournalEntry.user_id == current_user.id).all()

    return entries

@router.get("/{journal_id}", response_model=JournalEntryResponse)
def get_journal(
        journal_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    entry = get_user_journal(
        db=db,
        entry_id=journal_id,
        user_id=current_user.id,
    )

    return entry


@router.put("/{journal_id}", response_model=JournalEntryResponse)
def update_journal(
        journal_id: int,
        entry_update: JournalEntryUpdate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    entry = get_user_journal(
        db=db,
        entry_id=journal_id,
        user_id=current_user.id,
    )
    update_data = entry_update.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        setattr(entry, key, value)

    db.commit()
    db.refresh(entry)

    return entry

@router.delete("/{journal_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_journal(
        journal_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    entry = get_user_journal(
        db=db,
        entry_id=journal_id,
        user_id=current_user.id,
    )
    db.delete(entry)
    db.commit()

    return

@router.get(
    "/{journal_id}/emotions",
    response_model=list[EmotionAnalysisResponse]
)
def get_emotion_analysis(
        journal_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    entry = get_user_journal(
        db=db,
        entry_id=journal_id,
        user_id=current_user.id,
    )

    if entry.analysis_status in ["pending", "processing"]:
        raise HTTPException(status_code=status.HTTP_202_ACCEPTED, detail="Emotion analysis is still processing")

    emotions = (
        db.query(EmotionAnalysis)
        .filter(
            EmotionAnalysis.journal_entry_id == journal_id
        )
        .all()
    )

    return emotions

