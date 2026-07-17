from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.journal_entry import JournalEntry



def get_user_journal(
        db: Session,
        entry_id: int,
        user_id: int,
) -> JournalEntry:

    journal = (
        db.query(JournalEntry)
        .filter(JournalEntry.id == entry_id)
        .first()
    )

    if journal is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Journal not found")

    if journal.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized to view this journal"
        )

    return journal