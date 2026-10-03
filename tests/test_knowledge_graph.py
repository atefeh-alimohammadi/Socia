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



from app.database.database import SessionLocal
from app.services.knowledge_graph_service import (
    get_related_entities_for_tag,
)

db = SessionLocal()

try:
    results = get_related_entities_for_tag(
    db=db,
    user_id=4,
    tag="fear_of_judgment",
)

    print(results)

finally:
    db.close()