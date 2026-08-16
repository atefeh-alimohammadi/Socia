from sqlalchemy.orm import Session

from app.models.message_feedback import MessageFeedback
from app.models.journey_challenge import JourneyChallenge
from app.models.journey import Journey


MIN_FEEDBACK_SIGNALS = 3
MAX_RECENT_FEEDBACK = 10


def get_communication_policy(
    db: Session,
    user_id: int
) -> dict:

    message_feedback = (
        db.query(MessageFeedback)
        .filter(MessageFeedback.user_id == user_id)
        .order_by(MessageFeedback.id.desc())
        .limit(MAX_RECENT_FEEDBACK)
        .all()
    )

    journey_feedback = (
        db.query(JourneyChallenge)
        .join(
            Journey,
            Journey.id == JourneyChallenge.journey_id
        )
        .filter(
            Journey.user_id == user_id,
            JourneyChallenge.status.in_(["completed", "skipped"]),
        )
        .order_by(JourneyChallenge.created_at.desc())
        .limit(MAX_RECENT_FEEDBACK)
        .all()
    )

    signals = []

    for feedback in message_feedback:
        signals.append(feedback.rating)

    for challenge in journey_feedback:
        if challenge.difficulty_feedback:
            signals.append(challenge.difficulty_feedback)

        if challenge.skip_reason:
            signals.append(challenge.skip_reason)

    if len(signals) < MIN_FEEDBACK_SIGNALS:
        return {
            "pacing": "standard"
        }

    gentle_signals = {
        "not_helpful",
        "too_hard",
        "too_anxious",
        "too_difficult",
    }

    direct_signals = {
        "too_easy",
    }

    gentle_count = sum(
        1
        for signal in signals
        if signal in gentle_signals
    )

    direct_count = sum(
        1
        for signal in signals
        if signal in direct_signals
    )

    if gentle_count > direct_count:
        return {
            "pacing": "gentle"
        }

    if direct_count > gentle_count:
        return {
            "pacing": "direct"
        }

    return {
        "pacing": "standard"
    }