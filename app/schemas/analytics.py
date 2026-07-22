from pydantic import BaseModel


class EmotionAnalyticsResponse(BaseModel):
    total_entries: int
    emotion_counts: dict[str, int]
    top_emotion: str | None
    average_confidence: float