from fastapi import FastAPI
from app.api import auth, journal
from app.database.database import Base, engine
from app.models.user import User
from app.models.journal_entry import JournalEntry

from app.api.deps import get_current_user
from app.schemas.users import UserResponse
from fastapi import Depends

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(auth.router)
app.include_router(journal.router)
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/me", response_model=UserResponse)
def read_current_user(
        current_user: User = Depends(get_current_user)
):
    return current_user