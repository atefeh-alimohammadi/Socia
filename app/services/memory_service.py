import json
import logging

from ollama import chat

from app.database.database import SessionLocal

from app.models.journal_entry import JournalEntry
from app.models.emotion_analysis import EmotionAnalysis
from app.models.user_memory import UserMemory


logger = logging.getLogger(__name__)


MEMORY_SYNTHESIS_PROMPT = """
You are analyzing a user's emotional history and journal entries.

Your task is to identify meaningful long-term memories about this user.

Return ONLY valid JSON.
Do not add markdown.
Do not add explanations.

The JSON format must be exactly:

{
  "memories": [
    {
      "memory_type": "pattern",
      "tag": "overthinking",
      "content": "User often overthinks after social interactions.",
      "source": "journal"
    }
  ]
}


Rules:

- Return between 3 and 5 memories.
- memory_type must be one of:
  - pattern
  - preference
  - insight

- tag should be a short category label.
  Examples:
  - anxiety
  - confidence
  - overthinking
  - social_fear
  - motivation

- Keep content short, specific, and meaningful.
- Do not repeat similar memories.

Pattern examples:
- User often worries about being judged by others.
- User avoids speaking in groups because of fear.

Preference examples:
- User prefers small achievable challenges.

Insight examples:
- User shows willingness to improve despite difficulties.

Only return JSON.
"""


def synthesize_user_patterns(
        user_id: int,
) -> None:

    db = SessionLocal()

    try:

        journals = (
            db.query(JournalEntry)
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



        user_history = ""


        # Include journal text
        for journal in journals:

            user_history += (
                "Journal Entry:\n"
                f"{journal.content}\n\n"
            )


        # Include emotion analysis
        user_history += "\nEmotion Analysis:\n"


        for emotion in emotion_rows:

            user_history += (
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
                    "content": user_history
                }
            ]
        )


        ai_output = response.message.content


        data = json.loads(ai_output)


        memories = data.get(
            "memories",
            []
        )



        for item in memories:


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

                    source=item.get(
                        "source",
                        "journal"
                    ),

                    tag=item.get(
                        "tag"
                    )

                )


                db.add(memory)



        db.commit()



    except Exception as e:

        logger.error(
            "Memory synthesis failed: %s",
            e
        )

        db.rollback()



    finally:

        db.close()