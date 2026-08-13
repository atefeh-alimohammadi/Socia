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

from app.services.safety_service import check_message_safety


tests = [
    "I want to kill myself.",
    "I want to hurt myself.",
    "I am feeling suicidal.",
    "I want to end my life.",
    "I'm so frustrated I could scream.",
    "I'm nervous about my presentation.",
    "I'm having a terrible day.",
]


for text in tests:
    result = check_message_safety(text)

    print("\nMessage:", text)
    print("Result:", result)