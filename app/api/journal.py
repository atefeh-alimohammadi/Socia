from fastapi import APIRouter, Depends, HTTPException, status
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


router = APIRouter(
    prefix="/journal",
    tags=["journal"],
)

@router.post("/", response_model=JournalEntryResponse, status_code=status.HTTP_201_CREATED)
def create_journal_entry(
        entry: JournalEntryCreate,
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
    entry = db.query(JournalEntry).filter(JournalEntry.id == journal_id).first()

    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Journal not found")

    if entry.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    return entry


@router.put("/{journal_id}", response_model=JournalEntryResponse)
def update_journal(
        journal_id: int,
        entry_update: JournalEntryUpdate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    entry = db.query(JournalEntry).filter(JournalEntry.id == journal_id).first()

    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Journal not found")

    if entry.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

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

    entry = db.query(JournalEntry).filter(JournalEntry.id == journal_id).first()

    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Journal not found")

    if entry.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    db.delete(entry)
    db.commit()

    return

