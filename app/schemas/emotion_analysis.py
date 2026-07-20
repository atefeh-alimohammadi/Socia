from datetime import datetime

from pydantic import BaseModel, ConfigDict

class EmotionAnalysisResponse(BaseModel):
    id: int
    journal_entry_id: int
    emotion: str
    confidence_score: float
    notes: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)