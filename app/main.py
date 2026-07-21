from fastapi import FastAPI
from app.api import auth, journal
from app.database.database import Base, engine
from app.models import conversation_session
from app.models.user import User
from app.models.journal_entry import JournalEntry
from app.models.conversation_session import ConversationSession
from app.models.message import Message
from app.models.user_memory import UserMemory

from app.api.deps import get_current_user
from app.schemas.users import UserResponse
from fastapi import Depends

from app.api import conversation, auth, journal, memory



app = FastAPI()
app.include_router(auth.router)
app.include_router(journal.router)
app.include_router(conversation.router)
app.include_router(memory.router)
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/me", response_model=UserResponse)
def read_current_user(
        current_user: User = Depends(get_current_user)
):
    return current_user