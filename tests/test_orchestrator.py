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


from app.services.orchestrator import route_message


print(route_message("ok", False))

print(
    route_message(
        "I've been feeling anxious about speaking in meetings.",
        False,
    )
)

print(route_message("thanks", False))