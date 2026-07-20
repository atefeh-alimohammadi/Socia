import json

from sqlalchemy.orm import Session
from ollama import chat

from app.database.database import SessionLocal
from app.models.journal_entry import JournalEntry
from app.models.emotion_analysis import EmotionAnalysis

EMOTION_ANALYSIS_PROMPT = """
Analyze the following journal entry and identify the emotions present.

Return ONLY valid JSON.
Do not add markdown.
Do not add explanations.

The JSON format must be exactly:

{
  "emotions": [
    {
      "emotion": "anxiety",
      "confidence_score": 0.85,
      "notes": "User expresses worry about future"
    }
  ]
}

Rules:
- confidence_score must be between 0.0 and 1.0
- emotion should be a short emotion name
- notes should briefly explain why this emotion was detected

Journal entry:
"""

def analyze_emotions(
        journal_entry_id: int,
        content: str,
):
    db = SessionLocal()
    try:
        entry = (
        db.query(JournalEntry)
        .filter(JournalEntry.id == journal_entry_id)
        .first()
         )

        if not entry:
          return

        entry.analysis_status = "processing"
        db.commit()

        response = chat(
            model= "qwen2.5:7b",
            messages=[
                {
                    "role": "system",
                    "content": EMOTION_ANALYSIS_PROMPT
                },
                {
                    "role": "user",
                    "content": content
            }
            ]
        )
        ai_output = response.message.content

        data = json.loads(ai_output)

        emotions = data.get("emotions", [])

        for item in emotions:
            emotion = EmotionAnalysis(
                journal_entry_id=journal_entry_id,
                emotion=item["emotion"],
                confidence_score=item["confidence_score"],
                notes=item.get("notes")
            )
            db.add(emotion)
        entry.analysis_status = "completed"

        db.commit()

    except Exception as e:

        print("Emotion analysis error:", e)

        entry.analysis_status = "failed"

        db.commit()

    finally:
        db.close()


