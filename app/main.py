from fastapi import FastAPI
from app.api import auth
from app.database.database import Base, engine
from app.models.user import User
Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(auth.router)
@app.get("/")
async def root():
    return {"message": "Hello World"}