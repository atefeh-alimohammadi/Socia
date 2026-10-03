from app.models.user import User

from app.models.journal_entry import JournalEntry
from app.models.conversation_session import ConversationSession
from app.models.message import Message
from app.models.emotion_analysis import EmotionAnalysis

from app.models.conversation_emotion import ConversationEmotion
from app.models.behavior_observation import BehaviorObservation

from app.models.journey import Journey
from app.models.journey_challenge import JourneyChallenge

from app.models.user_memory import UserMemory
from app.models.episodic_memory import EpisodicMemory
from app.models.user_memory_history import UserMemoryHistory
from app.models.user_memory_evidence import UserMemoryEvidence
from app.database.database import SessionLocal
from app.services.episodic_memory_service import (
    retrieve_relevant_episodic_memories,
)
from app.database.database import SessionLocal
from app.services.episodic_memory_service import (
    retrieve_relevant_episodic_memories,
)


def test_retrieval(
    user_id: int,
    query_text: str,
    label: str,
):
    db = SessionLocal()

    try:
        results = retrieve_relevant_episodic_memories(
            db=db,
            user_id=user_id,
            query_text=query_text,
            limit=5,
        )

        print(f"\n=== Query: {label} ===")
        print(f'"{query_text}"\n')

        if not results:
            print("No relevant memories found.")
            return

        seen_messages = set()

        for memory in results:
            print(
                f"id={memory.id} | "
                f"msg={memory.message_id} | "
                f"{memory.event_type}/{memory.tag} | "
                f"{memory.content}"
            )

            if memory.message_id in seen_messages:
                print(
                    f"!!! DUPLICATE MESSAGE_ID FOUND: "
                    f"{memory.message_id}"
                )

            seen_messages.add(memory.message_id)

        print(
            f"\nTotal memories: {len(results)}"
        )

        print(
            f"Unique source messages: "
            f"{len(seen_messages)}"
        )

    finally:
        db.close()


# ============================================================
# TEST 1 — Relevant query
# ============================================================

test_retrieval(
    4,
    "I'm afraid that people will judge me when I speak in front of others.",
    "relevant fear of judgment",
)


# ============================================================
# TEST 2 — Unrelated query
# ============================================================

test_retrieval(
    4,
    "I'm trying to learn Python and improve my programming skills.",
    "unrelated Python",
)


# ============================================================
# TEST 3 — Conflict avoidance
# ============================================================

test_retrieval(
    4,
    "I avoided confronting my friend because I didn't want to make the situation worse.",
    "conflict avoidance",
)