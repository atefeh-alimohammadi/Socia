from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.deps import get_db
from app.api.deps import get_current_user

from app.models.user import User
from app.models.emotion_analysis import EmotionAnalysis
from app.models.journal_entry import JournalEntry

from app.schemas.analytics import EmotionAnalyticsResponse

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
)

@router.get(
    "/emotions",
    response_model=EmotionAnalyticsResponse,
)
def get_emotion_analytics(
    db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):
    emotions = (
        db.query(EmotionAnalysis)
        .join(JournalEntry)
        .filter(JournalEntry.user_id == current_user.id)
        .all()
    )

    total_entries = (
        db.query(JournalEntry)
        .filter(JournalEntry.user_id == current_user.id)
        .count()
    )

    emotion_counts = Counter()

    confidence_sum = 0.0

    for emotion in emotions:
        emotion_counts[emotion.emotion] += 1
        confidence_sum += emotion.confidence_score

    top_emotion = None

    if emotion_counts:
        top_emotion = emotion_counts.most_common(1)[0][0]

    average_confidence = 0.0

    if emotions:
        average_confidence = confidence_sum / len(emotions)

    return EmotionAnalyticsResponse(
        total_entries=total_entries,
        emotion_counts=dict(emotion_counts),
        top_emotion=top_emotion,
        average_confidence=round(average_confidence, 2),

    )


