from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UserMemoryResponse(BaseModel):
    __tablename__ = "user_memory"
    id: int
    user_id: int
    memory_type: str
    content: str
    source: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
