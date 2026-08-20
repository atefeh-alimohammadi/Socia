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
    source_memory_id: int | None
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
    skip_reason: str | None
    skip_reason_detail: str | None
    difficulty_feedback: str | None
    emotional_response: str | None

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


class JourneyFromPatternRequest(BaseModel):
    memory_id: int


class ChallengeCompleteRequest(BaseModel):
    difficulty_feedback: str | None = Field(
        default=None,
        pattern="^(too_easy|just_right|too_hard)$",
    )

    emotional_response: str | None = None


class ChallengeSkipRequest(BaseModel):
    skip_reason: str = Field(
        pattern="^(busy|forgot|too_difficult|too_anxious|not_relevant|disliked_activity|situation_unavailable|other)$",
    )

    skip_reason_detail: str | None = None