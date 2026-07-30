from sqlalchemy.orm import Session

from app.models.user_memory import UserMemory
from app.models.journal_entry import JournalEntry
from app.models.emotion_analysis import EmotionAnalysis
from app.models.journey import Journey


def assemble_user_context(
        user_id: int,
        db: Session
) -> dict:

    memories = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == user_id)
        .all()
    )

    emotions = (
        db.query(EmotionAnalysis)
        .join(JournalEntry)
        .filter(JournalEntry.user_id == user_id)
        .order_by(EmotionAnalysis.created_at.desc())
        .limit(10)
        .all()
    )

    active_journeys = (
        db.query(Journey)
        .filter(
            Journey.user_id == user_id,
            Journey.status == "active"
        )
        .all()
    )


    return {
        "memories": memories,
        "recent_emotions": emotions,
        "active_journeys": [
            journey.title
            for journey in active_journeys
        ]
    }



def format_context_for_prompt(
        context: dict
) -> str:

    text = "What Socia knows about this user:\n"


    for memory in context["memories"]:
        label = memory.tag if memory.tag else memory.memory_type
        text += f"- {label}: {memory.content}\n"


    if context["recent_emotions"]:

        text += "\nRecent emotions:\n"

        for emotion in context["recent_emotions"]:
            text += (
                f"- {emotion.emotion}: "
                f"{emotion.confidence_score}\n"
            )


    if context["active_journeys"]:

        text += "\nCurrently working on:\n"

        for journey in context["active_journeys"]:
            text += f"- {journey}\n"


    return text