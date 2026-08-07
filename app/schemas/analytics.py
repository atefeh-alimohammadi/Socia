from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EmotionAnalyticsResponse(BaseModel):
    total_entries: int
    emotion_counts: dict[str, int]
    top_emotion: str | None
    average_confidence: float



class EmotionSeriesItem(BaseModel):
    emotion: str
    color: str
    data: list[float | None]

class EmotionTimelineResponse(BaseModel):
    weeks: list[str]
    series: list[EmotionSeriesItem]


class BehaviorPatternResponse(BaseModel):
    id: int
    tag: str
    content: str
    evidence_count: int
    confidence: float| None
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )

