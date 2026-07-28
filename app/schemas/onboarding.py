from pydantic import BaseModel, Field


class OnboardingAnswerCreate(BaseModel):
    step: int = Field(ge=1, le=5)
    answer: str = Field(min_length=1)



class OnboardingCompleteResponse(BaseModel):
    summary: str
    onboarding_completed: bool