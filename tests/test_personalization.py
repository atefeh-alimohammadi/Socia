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


from app.models.user import User
from app.models.conversation_session import ConversationSession
from app.models.message import Message
from app.models.message_feedback import MessageFeedback
from app.models.journey import Journey
from app.models.journey_challenge import JourneyChallenge

from app.database.database import SessionLocal
from app.services.personalization_service import get_communication_policy


def create_test_user(db, username):
    email = f"{username}@test.com"

    # Reuse existing test user if it already exists.
    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if user:
        return user

    user = User(
        email=email,
        username=username,
        hashed_password="test_password_hash",
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def create_message_feedback(db, user, rating):
    # Create a conversation session belonging to the user.
    session = ConversationSession(
        user_id=user.id,
        title="Personalization Test Session",
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    # Create an assistant message belonging to that session.
    message = Message(
        session_id=session.id,
        role="assistant",
        content="Test assistant message",
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    # Create feedback for the assistant message.
    feedback = MessageFeedback(
        message_id=message.id,
        user_id=user.id,
        rating=rating,
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)


def create_journey_feedback(
    db,
    user,
    difficulty=None,
    skip_reason=None,
):
    journey = Journey(
        user_id=user.id,
        title="Personalization Test Journey",
        description="Temporary personalization test",
        source="user_created",
        status="active",
        day_current=1,
        day_total=1,
    )

    db.add(journey)
    db.commit()
    db.refresh(journey)

    challenge = JourneyChallenge(
        journey_id=journey.id,
        day_number=1,
        title="Test Challenge",
        description="Temporary challenge",
        status="skipped" if skip_reason else "completed",
        difficulty_feedback=difficulty,
        skip_reason=skip_reason,
    )

    db.add(challenge)
    db.commit()
    db.refresh(challenge)


def run_test():
    db = SessionLocal()

    test_users = []

    try:

        # =========================================================
        # TEST 1: Insufficient feedback -> standard
        # =========================================================

        user_standard = create_test_user(
            db,
            "personalization_standard_test",
        )

        test_users.append(user_standard)

        # Make sure this test user starts clean.
        db.query(MessageFeedback).filter(
            MessageFeedback.user_id == user_standard.id
        ).delete(synchronize_session=False)

        db.query(JourneyChallenge).filter(
            JourneyChallenge.journey.has(
                Journey.user_id == user_standard.id
            )
        ).delete(synchronize_session=False)

        db.commit()

        create_message_feedback(
            db,
            user_standard,
            "helpful",
        )

        policy = get_communication_policy(
            db,
            user_standard.id,
        )

        print("\n=== TEST 1: INSUFFICIENT FEEDBACK ===")
        print("Expected: standard")
        print("Actual:  ", policy)

        assert policy["pacing"] == "standard"


        # =========================================================
        # TEST 2: Gentle pattern -> gentle
        # =========================================================

        user_gentle = create_test_user(
            db,
            "personalization_gentle_test",
        )

        test_users.append(user_gentle)

        # Clean previous test data for this user.
        db.query(MessageFeedback).filter(
            MessageFeedback.user_id == user_gentle.id
        ).delete(synchronize_session=False)

        db.query(JourneyChallenge).filter(
            JourneyChallenge.journey.has(
                Journey.user_id == user_gentle.id
            )
        ).delete(synchronize_session=False)

        db.commit()

        create_message_feedback(
            db,
            user_gentle,
            "not_helpful",
        )

        create_journey_feedback(
            db,
            user_gentle,
            difficulty="too_hard",
        )

        create_journey_feedback(
            db,
            user_gentle,
            skip_reason="too_anxious",
        )

        policy = get_communication_policy(
            db,
            user_gentle.id,
        )

        print("\n=== TEST 2: GENTLE PATTERN ===")
        print("Expected: gentle")
        print("Actual:  ", policy)

        assert policy["pacing"] == "gentle"


        # =========================================================
        # TEST 3: Direct pattern -> direct
        # =========================================================

        user_direct = create_test_user(
            db,
            "personalization_direct_test",
        )

        test_users.append(user_direct)

        # Clean previous test data for this user.
        db.query(MessageFeedback).filter(
            MessageFeedback.user_id == user_direct.id
        ).delete(synchronize_session=False)

        db.query(JourneyChallenge).filter(
            JourneyChallenge.journey.has(
                Journey.user_id == user_direct.id
            )
        ).delete(synchronize_session=False)

        db.commit()

        create_message_feedback(
            db,
            user_direct,
            "helpful",
        )

        create_journey_feedback(
            db,
            user_direct,
            difficulty="too_easy",
        )

        create_journey_feedback(
            db,
            user_direct,
            difficulty="too_easy",
        )

        policy = get_communication_policy(
            db,
            user_direct.id,
        )

        print("\n=== TEST 3: DIRECT PATTERN ===")
        print("Expected: direct")
        print("Actual:  ", policy)

        assert policy["pacing"] == "direct"


        # =========================================================
        # ALL TESTS PASSED
        # =========================================================

        print("\n===================================")
        print("ALL PERSONALIZATION POLICY TESTS PASSED")
        print("===================================")

    finally:

        # We intentionally keep the test users/data for now.
        # This makes it easier to inspect the database after the test.
        db.close()


if __name__ == "__main__":
    run_test()