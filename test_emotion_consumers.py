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


from app.database.database import SessionLocal

from app.models.user import User
from app.models.conversation_session import ConversationSession
from app.models.message import Message
from app.models.conversation_emotion import ConversationEmotion

from app.services.emotion_analytics_service import get_emotion_trend


def run_test():

    db = SessionLocal()

    test_user = None
    test_session = None
    test_message = None
    test_emotion = None

    try:

        # ---------------------------------------------------------
        # Create isolated test user
        # ---------------------------------------------------------

        test_user = User(
            email="emotion_consumer_test@test.com",
            username="emotion_consumer_test",
            hashed_password="test_password_hash",
        )

        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        # ---------------------------------------------------------
        # Create test conversation session
        # ---------------------------------------------------------

        test_session = ConversationSession(
            user_id=test_user.id,
            title="Emotion Consumer Test",
        )

        db.add(test_session)
        db.commit()
        db.refresh(test_session)

        # ---------------------------------------------------------
        # Create test message
        # ---------------------------------------------------------

        test_message = Message(
            session_id=test_session.id,
            role="user",
            content="I felt anxious but hopeful about the presentation.",
        )

        db.add(test_message)
        db.commit()
        db.refresh(test_message)

        # ---------------------------------------------------------
        # Create ConversationEmotion with dimensional values
        # ---------------------------------------------------------

        test_emotion = ConversationEmotion(
            message_id=test_message.id,
            user_id=test_user.id,
            emotion="anxiety",
            intensity=0.8,
            valence=-0.6,
            arousal=0.8,
        )

        db.add(test_emotion)
        db.commit()
        db.refresh(test_emotion)

        # ---------------------------------------------------------
        # TEST 1
        # Existing emotion fields still work
        # ---------------------------------------------------------

        emotion = (
            db.query(ConversationEmotion)
            .filter(
                ConversationEmotion.id == test_emotion.id
            )
            .first()
        )

        assert emotion is not None
        assert emotion.emotion == "anxiety"
        assert emotion.intensity == 0.8

        print("\n=== TEST 1: EXISTING EMOTION FIELDS ===")
        print("Emotion:", emotion.emotion)
        print("Intensity:", emotion.intensity)
        print("PASS")

        # ---------------------------------------------------------
        # TEST 2
        # New dimensional fields work
        # ---------------------------------------------------------

        assert emotion.valence == -0.6
        assert emotion.arousal == 0.8

        print("\n=== TEST 2: DIMENSIONAL FIELDS ===")
        print("Valence:", emotion.valence)
        print("Arousal:", emotion.arousal)
        print("PASS")

        # ---------------------------------------------------------
        # TEST 3
        # Trend analytics works with the same row
        # ---------------------------------------------------------

        result = get_emotion_trend(
            db=db,
            user_id=test_user.id,
            days=14,
        )

        assert result["average_valence"] == -0.6
        assert result["average_arousal"] == 0.8

        print("\n=== TEST 3: EMOTION ANALYTICS ===")
        print(
            "Average valence:",
            result["average_valence"]
        )
        print(
            "Average arousal:",
            result["average_arousal"]
        )
        print("PASS")

        # ---------------------------------------------------------
        # TEST 4
        # Relationship to Message still works
        # ---------------------------------------------------------

        message_emotions = test_message.conversation_emotions

        assert len(message_emotions) == 1
        assert message_emotions[0].id == test_emotion.id

        print("\n=== TEST 4: MESSAGE RELATIONSHIP ===")
        print(
            "ConversationEmotion rows:",
            len(message_emotions)
        )
        print("PASS")

        # ---------------------------------------------------------
        # TEST 5
        # Relationship to User still works
        # ---------------------------------------------------------

        user_emotions = test_user.conversation_emotions

        assert any(
            emotion.id == test_emotion.id
            for emotion in user_emotions
        )

        print("\n=== TEST 5: USER RELATIONSHIP ===")
        print(
            "User emotion relationship: OK"
        )
        print("PASS")

        # ---------------------------------------------------------
        # Final result
        # ---------------------------------------------------------

        print("\n===================================")
        print("ALL EMOTION CONSUMER TESTS PASSED")
        print("===================================")

    finally:

        # ---------------------------------------------------------
        # Cleanup
        # ---------------------------------------------------------

        if test_emotion is not None:
            db.delete(test_emotion)
            db.commit()

        if test_message is not None:
            db.delete(test_message)
            db.commit()

        if test_session is not None:
            db.delete(test_session)
            db.commit()

        if test_user is not None:
            db.delete(test_user)
            db.commit()

        db.close()


if __name__ == "__main__":
    run_test()