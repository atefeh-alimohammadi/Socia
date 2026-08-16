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

from app.models.user import User
from app.models.conversation_emotion import ConversationEmotion

from app.database.database import SessionLocal
from app.services.emotion_analytics_service import get_emotion_trend


def run_test():

    db = SessionLocal()

    test_user = None
    test_emotions = []

    try:

        # ---------------------------------------------------------
        # Create isolated test user
        # ---------------------------------------------------------

        test_user = User(
            email="emotion_trend_test@test.com",
            username="emotion_trend_test",
            hashed_password="test_password_hash",
        )

        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        # ---------------------------------------------------------
        # Seed known dimensional values
        # ---------------------------------------------------------

        values = [
            (-0.8, 0.9),
            (-0.4, 0.7),
            (0.6, 0.5),
            (0.2, 0.3),
        ]

        for valence, arousal in values:

            emotion = ConversationEmotion(
                message_id=142,
                user_id=test_user.id,
                emotion="anxiety",
                intensity=0.8,
                valence=valence,
                arousal=arousal,
            )

            db.add(emotion)
            test_emotions.append(emotion)

        db.commit()

        # ---------------------------------------------------------
        # Calculate expected averages
        # ---------------------------------------------------------

        expected_valence = (
            -0.8
            - 0.4
            + 0.6
            + 0.2
        ) / 4

        expected_arousal = (
            0.9
            + 0.7
            + 0.5
            + 0.3
        ) / 4

        # ---------------------------------------------------------
        # Query trend
        # ---------------------------------------------------------

        result = get_emotion_trend(
            db=db,
            user_id=test_user.id,
            days=14,
        )

        print("\n=== TEST 3: EMOTION TREND ===")

        print(
            "Expected average valence:",
            expected_valence
        )

        print(
            "Actual average valence:",
            result["average_valence"]
        )

        print(
            "Expected average arousal:",
            expected_arousal
        )

        print(
            "Actual average arousal:",
            result["average_arousal"]
        )

        # ---------------------------------------------------------
        # Assertions
        # ---------------------------------------------------------

        assert result["average_valence"] is not None

        assert result["average_arousal"] is not None

        assert abs(
            result["average_valence"] - expected_valence
        ) < 0.0001

        assert abs(
            result["average_arousal"] - expected_arousal
        ) < 0.0001

        print("\n===================================")
        print("TEST 3 PASSED")
        print("===================================")

    finally:

        # ---------------------------------------------------------
        # Cleanup test emotions
        # ---------------------------------------------------------

        for emotion in test_emotions:

            try:
                db.delete(emotion)

            except Exception:
                pass

        db.commit()

        # ---------------------------------------------------------
        # Cleanup test user
        # ---------------------------------------------------------

        if test_user is not None:

            try:
                db.delete(test_user)
                db.commit()

            except Exception:
                db.rollback()

        db.close()


if __name__ == "__main__":
    run_test()