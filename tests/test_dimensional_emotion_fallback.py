from app.models.user import User
from app.models.journal_entry import JournalEntry
from app.models.conversation_session import ConversationSession
from app.models.message import Message
from app.models.user_memory import UserMemory
from app.models.journey import Journey
from app.models.journey_challenge import JourneyChallenge
from app.models.emotion_analysis import EmotionAnalysis
from app.models.conversation_emotion import ConversationEmotion
from app.models.behavior_observation import BehaviorObservation
from app.models.user_memory_history import UserMemoryHistory
from app.models.user_memory_evidence import UserMemoryEvidence
from app.models.episodic_memory import EpisodicMemory
from app.models.message_feedback import MessageFeedback
from app.models.knowledge_entity import KnowledgeEntity
from app.models.knowledge_edge import KnowledgeEdge
from app.models.user_memory_evidence import UserMemoryEvidence

import json

from app.database.database import SessionLocal
from app.models.conversation_session import ConversationSession
from app.models.message import Message
from app.models.conversation_emotion import ConversationEmotion
from app.models.episodic_memory import EpisodicMemory

import app.services.conversation_analyzer as analyzer


TEST_USER_ID = 4


def fake_generate_ai_response(prompt: str):
    """
    Simulate an LLM response where dimensional emotion
    values are invalid.
    """

    return json.dumps(
        {
            "emotions": [
                {
                    "emotion": "anxiety",
                    "intensity": 0.8,
                    "valence": 5,
                    "arousal": -1,
                    "evidence": (
                        "I was extremely worried about what "
                        "might happen."
                    ),
                }
            ],
            "observations": [],
            "entities": [],
        }
    )


def run_test():

    db = SessionLocal()

    session = None
    message = None

    try:

        # ---------------------------------------------------------
        # Create temporary conversation session
        # ---------------------------------------------------------

        session = ConversationSession(
            user_id=TEST_USER_ID,
            title="Dimensional Emotion Fallback Test",
        )

        db.add(session)
        db.commit()
        db.refresh(session)

        # ---------------------------------------------------------
        # Create temporary user message
        # ---------------------------------------------------------

        message = Message(
            session_id=session.id,
            role="user",
            content=(
                "I was extremely worried about what "
                "might happen."
            ),
        )

        db.add(message)
        db.commit()
        db.refresh(message)

        # ---------------------------------------------------------
        # Replace the real LLM call with deterministic response
        # ---------------------------------------------------------

        original_generate_ai_response = (
            analyzer.generate_ai_response
        )

        analyzer.generate_ai_response = fake_generate_ai_response

        try:

            analyzer.analyze_conversation_message(
                message_id=message.id,
                user_id=TEST_USER_ID,
                session_id=session.id,
                content=message.content,
            )

        finally:

            analyzer.generate_ai_response = (
                original_generate_ai_response
            )

        # ---------------------------------------------------------
        # Check generated emotion
        # ---------------------------------------------------------

        emotion = (
            db.query(ConversationEmotion)
            .filter(
                ConversationEmotion.message_id == message.id
            )
            .first()
        )

        print("\n=== TEST 2: INVALID DIMENSIONAL VALUES ===")

        assert emotion is not None, (
            "Emotion was not created."
        )

        print(
            "Emotion:",
            emotion.emotion
        )

        print(
            "Intensity:",
            emotion.intensity
        )

        print(
            "Valence:",
            emotion.valence
        )

        print(
            "Arousal:",
            emotion.arousal
        )

        # ---------------------------------------------------------
        # Required assertions
        # ---------------------------------------------------------

        assert emotion.emotion == "anxiety"

        assert emotion.intensity == 0.8

        assert emotion.valence is None, (
            "Invalid valence should be stored as NULL."
        )

        assert emotion.arousal is None, (
            "Invalid arousal should be stored as NULL."
        )

        print("\nExpected:")
        print("Emotion is preserved: anxiety")
        print("Intensity is preserved: 0.8")
        print("Invalid valence -> NULL")
        print("Invalid arousal -> NULL")

        print("\nActual:")
        print(
            f"emotion={emotion.emotion}, "
            f"intensity={emotion.intensity}, "
            f"valence={emotion.valence}, "
            f"arousal={emotion.arousal}"
        )

        print("\n===================================")
        print("TEST 2 PASSED")
        print("===================================")


    finally:

        # ---------------------------------------------------------

        # Cleanup test data only

        # ---------------------------------------------------------

        if message is not None:

            db.query(ConversationEmotion).filter(

                ConversationEmotion.message_id == message.id

            ).delete(

                synchronize_session=False

            )

            episodic_memories = (

                db.query(EpisodicMemory)

                .filter(

                    EpisodicMemory.message_id == message.id

                )

                .all()

            )

            for memory in episodic_memories:
                db.query(UserMemoryEvidence).filter(

                    UserMemoryEvidence.episodic_memory_id == memory.id

                ).delete(

                    synchronize_session=False

                )

                db.delete(memory)

            db.delete(message)

        if session is not None:
            db.delete(session)

        db.commit()

        db.close()

if __name__ == "__main__":
    run_test()