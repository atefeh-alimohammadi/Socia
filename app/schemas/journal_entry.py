from datetime import datetime
from pydantic import BaseModel

class JournalEntryCreate(BaseModel):
    title: str
    content: str
    mood: str | None = None

class JournalEntryUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    mood: str | None = None

class JournalEntryResponse(BaseModel):
    id: int
    title: str
    content: str
    mood: str | None
    analysis_status: str
    user_id: int
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }