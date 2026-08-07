from app.models.user import User
from app.models.journal_entry import JournalEntry
from app.models.conversation_session import ConversationSession
from app.models.message import Message
from app.models.emotion_analysis import EmotionAnalysis
from app.models.user_memory import UserMemory

from app.services.memory_service import synthesize_patterns_from_observations


synthesize_patterns_from_observations(1)