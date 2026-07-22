import json

from ollama import chat

from app.database.database import SessionLocal

from app.models.journal_entry import JournalEntry
from app.models.emotion_analysis import EmotionAnalysis
from app.models.user_memory import UserMemory

import logging

logger = logging.getLogger(__name__)

MEMORY_SYNTHESIS_PROMPT = """
You are analyzing a user's emotional history.

Your task is to identify long-term patterns.

Return ONLY valid JSON.

Do not add markdown.
Do not add explanations.

The JSON format must be exactly:

{
  "patterns": [
    {
      "memory_type": "pattern",
      "content": "User frequently experiences anxiety in work-related situations.",
      "source": "journal"
    }
  ]
}

Rules:
- Return between 3 and 5 patterns.
- memory_type must be one of:
  - pattern
  - preference
  - insight
- Keep each content short and meaningful.
- Do not repeat similar patterns.
"""

def synthesize_user_patterns(
        user_id: int,
) -> None:
    db = SessionLocal()

    try:

        journals = (db.query(JournalEntry)
        .filter(
            JournalEntry.user_id == user_id,
            JournalEntry.analysis_status == "completed"
        )
        .all()
        )

        if not journals:
            return

        emotion_rows = (
            db.query(EmotionAnalysis)
            .join(JournalEntry)
            .filter(
                JournalEntry.user_id == user_id
            )
            .all()
        )

        if not emotion_rows:
            return

        emotion_history = ""

        for emotion in emotion_rows:
            emotion_history += (
                f"Emotion: {emotion.emotion}\n"
                f"Confidence: {emotion.confidence_score}\n"
                f"Notes: {emotion.notes}\n\n"
            )
        response = chat(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "system",
                    "content": MEMORY_SYNTHESIS_PROMPT
                },
                {
                    "role": "user",
                    "content": emotion_history
                }
            ]
        )

        ai_output = response.message.content

        data = json.loads(ai_output)

        patterns = data.get("patterns", [])

        for item in patterns:
            existing = (
                db.query(UserMemory)
                .filter(
                    UserMemory.user_id == user_id,
                    UserMemory.content == item["content"]
                )
                .first()
            )

            if not existing:
                memory = UserMemory(
                    user_id=user_id,
                    memory_type=item["memory_type"],
                    content=item["content"],
                    source=item.get("source")
                )

                db.add(memory)

        db.commit()


    except Exception as e:
        logger.error("Memory synthesis failed: %s ", e)

        db.rollback()

    finally:
        db.close()

