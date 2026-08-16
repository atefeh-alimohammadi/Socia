from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.conversation_emotion import ConversationEmotion


DEFAULT_WINDOW_DAYS = 14


def get_emotion_trend(
        db: Session,
        user_id: int,
        days: int = DEFAULT_WINDOW_DAYS
) -> dict:

    since = datetime.now(timezone.utc) - timedelta(days=days)

    result = (
        db.query(
            func.avg(ConversationEmotion.valence).label(
                "average_valence"
            ),
            func.avg(ConversationEmotion.arousal).label(
                "average_arousal"
            ),
        )
        .filter(
            ConversationEmotion.user_id == user_id,
            ConversationEmotion.created_at >= since,
        )
        .first()
    )

    return {
        "user_id": user_id,
        "days": days,
        "average_valence": (
            float(result.average_valence)
            if result.average_valence is not None
            else None
        ),
        "average_arousal": (
            float(result.average_arousal)
            if result.average_arousal is not None
            else None
        ),
    }