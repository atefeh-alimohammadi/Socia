from collections import Counter

from fastapi import APIRouter, Depends

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.deps import get_db
from app.api.deps import get_current_user

from app.models.user import User
from app.models.conversation_emotion import ConversationEmotion
from app.models.user_memory import UserMemory
from app.models.emotion_analysis import EmotionAnalysis
from app.models.journal_entry import JournalEntry

from app.services.emotion_analytics_service import get_emotion_trend
from app.services.knowledge_graph_service import get_related_entities_for_tag

from app.schemas.analytics import (
    EmotionAnalyticsResponse,
    EmotionTimelineResponse,
    BehaviorPatternResponse,
)

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
)

TRACKED_EMOTIONS = [
    "anxiety",
    "confidence",
    "fear",
    "excitement",
    "frustration",
    "sadness",
    "shame",
    "hope",
]

EMOTION_COLORS = {
    "anxiety": "#ef4444",
    "confidence": "#6366f1",
    "fear": "#f97316",
    "excitement": "#22c55e",
    "frustration": "#f59e0b",
    "sadness": "#64748b",
    "shame": "#a855f7",
    "hope": "#06b6d4",
}

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


@router.get(
    "/emotion-timeline",
    response_model=EmotionTimelineResponse,
)
def get_emotion_timeline(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    now = datetime.now(timezone.utc)

    week_ranges = []
    week_labels = []

    for i in range(7, -1, -1):

        week_start = now - timedelta(weeks=i + 1)
        week_end = now - timedelta(weeks=i)

        week_ranges.append(
            (
                week_start,
                week_end,
            )
        )

        week_labels.append(
            f"Week {8-i}"
        )


    series = []


    for emotion in TRACKED_EMOTIONS:

        data = []

        for week_start, week_end in week_ranges:

            avg = (
                db.query(
                    func.avg(
                        ConversationEmotion.intensity
                    )
                )
                .filter(
                    ConversationEmotion.user_id == current_user.id,
                    ConversationEmotion.emotion == emotion,
                    ConversationEmotion.created_at >= week_start,
                    ConversationEmotion.created_at < week_end,
                )
                .scalar()
            )


            value = (
                round(float(avg), 2)
                if avg is not None
                else None
            )

            data.append(value)


        series.append(
            {
                "emotion": emotion,
                "color": EMOTION_COLORS[emotion],
                "data": data,
            }
        )


    return {
        "weeks": week_labels,
        "series": series,
    }


@router.get(
    "/behavior-patterns",
    response_model=list[BehaviorPatternResponse],
)
def get_behavior_patterns(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    return (
        db.query(UserMemory)
        .filter(
            UserMemory.user_id == current_user.id,
            UserMemory.source_type == "conversation",
            UserMemory.memory_type == "pattern",
        )
        .order_by(
            UserMemory.evidence_count.desc()
        )
        .all()
    )


@router.get(
    "/emotion-trend",
)
def get_emotion_trend_route(
    days: int = 14,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_emotion_trend(
        db=db,
        user_id=current_user.id,
        days=days,
    )


@router.get(
    "/related-entities",
)
def get_related_entities_route(
    tag: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_related_entities_for_tag(
        db=db,
        user_id=current_user.id,
        tag=tag,
    )