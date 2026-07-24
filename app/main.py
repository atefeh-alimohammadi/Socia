from fastapi import FastAPI
from app.database.database import Base, engine

from app.models.user import User
from app.models.journal_entry import JournalEntry
from app.models.conversation_session import ConversationSession
from app.models.message import Message
from app.models.user_memory import UserMemory


from app.api.deps import get_current_user
from app.schemas.users import UserResponse
from fastapi import Depends

from app.api import conversation, auth, journal, memory, analytics

from app.core.logging_config import setup_logging

from fastapi.middleware.cors import CORSMiddleware


setup_logging()
app = FastAPI(
    title="Socia API",
    description="AI-powered social coaching and emotional awareness companion",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth.router, prefix="/api/v1")
app.include_router(journal.router, prefix="/api/v1")
app.include_router(conversation.router, prefix="/api/v1")
app.include_router(memory.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/me", response_model=UserResponse)
def read_current_user(
        current_user: User = Depends(get_current_user)
):
    return current_user

@app.get("/health")
async def health_check():
    return {"status": "healthy",
            "version": "0.1.0",
            "service": "Socia API"
            }