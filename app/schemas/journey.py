from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class JourneyCreate(BaseModel):
    title: str = Field(
        min_length=1,
        max_length=200,
    )

    description: str = Field(
        min_length=1,
    )


class JourneyResponse(BaseModel):
    id: int
    user_id: int
    title: str
    description: str
    source: str
    status: str
    day_current: int
    day_total: int
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )


class ChallengeResponse(BaseModel):
    id: int
    journey_id: int
    day_number: int
    title: str
    description: str
    status: str
    completed_at: datetime | None

    model_config = ConfigDict(
        from_attributes=True
    )


class JourneyDetailResponse(BaseModel):
    journey: JourneyResponse
    challenges: list[ChallengeResponse]
    today_challenge: ChallengeResponse | None

    model_config = ConfigDict(
        from_attributes=True
    )