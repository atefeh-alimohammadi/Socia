from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from app.database.deps import get_db

from app.models.user import User
from app.models.journey import Journey
from app.models.journey_challenge import JourneyChallenge
from app.models.user_memory import UserMemory

from app.schemas.journey import (
JourneyCreate,
JourneyResponse,
JourneyDetailResponse,
ChallengeResponse,
JourneyFromPatternRequest,
)

from app.utils.token import get_current_user

from app.services.context_assembler import (
assemble_user_context,
format_context_for_prompt
)

from app.services.journey_service import (
generate_journey_challenges
)

from fastapi import Body
from datetime import datetime, timezone
from typing import Optional

router = APIRouter(
    prefix="/journeys",
    tags=["Journeys"]
)


@router.post("/", response_model=JourneyDetailResponse)
def create_journey(
        data: JourneyCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):

    context = assemble_user_context(
        current_user.id,
        db
    )

    formatted_context = format_context_for_prompt(context)


    generated = generate_journey_challenges(
        title=data.title,
        description=data.description,
        user_context=formatted_context,
    )

    journey = Journey(
        user_id=current_user.id,
        title=data.title,
        description=data.description,
        source="user_created",
        status="active",
        day_current=1,
        day_total=generated["day_total"],
    )

    db.add(journey)
    db.commit()
    db.refresh(journey)

    for challenge in generated["challenges"]:
        db_challenge = JourneyChallenge(
            journey_id=journey.id,
            day_number=challenge["day_number"],
            title=challenge["title"],
            description=challenge["description"],
            status="pending"
        )

        db.add(db_challenge)

    db.commit()
    db.refresh(journey)

    return{
        "journey": journey,
        "challenges": journey.challenges,
        "today_challenge": journey.challenges[0]
    }

@router.get("/", response_model=list[JourneyResponse])
def read_journeys(
    status: Optional[str] = "active",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    query = (
        db.query(Journey)
        .filter(Journey.user_id == current_user.id)
    )

    if status:
        query = query.filter(Journey.status == status)

    return query.all()


@router.get("/{journey_id}", response_model=JourneyDetailResponse)
def get_journey(
        journey_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):

    journey = (
        db.query(Journey)
        .filter(Journey.id == journey_id, Journey.user_id == current_user.id)
        .first()
    )

    if not journey:
        raise HTTPException(status_code=404, detail="Journey not found")

    today = None

    for challenge in journey.challenges:
        if challenge.day_number == journey.day_current:
            today = challenge
            break

    return {
        "journey": journey,
        "challenges": journey.challenges,
        "today_challenge": today
    }

@router.post("/{journey_id}/complete-challenge", response_model=JourneyDetailResponse)
def complete_challenge(
        journey_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):
    journey = (
        db.query(Journey)
        .filter(
            Journey.id == journey_id,
            Journey.user_id == current_user.id
        )
        .first()
    )

    if not journey:
        raise HTTPException(status_code=404, detail="Journey not found")


    current_challenge = (
        db.query(JourneyChallenge)
        .filter(
            JourneyChallenge.journey_id == journey_id,
            JourneyChallenge.day_number == journey.day_current
        )
        .first()
    )

    if current_challenge:
        current_challenge.status = "completed"
        current_challenge.completed_at = datetime.now(timezone.utc)

    if journey.day_current < journey.day_total:
        journey.day_current += 1

    else:
        journey.status = "completed"

    db.commit()
    db.refresh(journey)

    today = (
        db.query(JourneyChallenge)
        .filter(
            JourneyChallenge.journey_id == journey_id,
            JourneyChallenge.day_number == journey.day_current
        )
        .first()
    )

    return {
        "journey": journey,
        "challenges": sorted(
            journey.challenges,
            key=lambda x: x.day_number
        ),
        "today_challenge": today
    }

@router.patch("/{journey_id}/status")
def update_journey_status(
        journey_id: int,
        status: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):

    journey = (
        db.query(Journey)
        .filter(
            Journey.id == journey_id,
            Journey.user_id == current_user.id
        )
        .first()
    )

    if not journey:
        raise HTTPException(status_code=404, detail="Journey not found")

    if status not in [
        "active",
        "paused",
        "completed"
    ]:
        raise HTTPException(status_code=400, detail="Invalid status")

    journey.status = status

    db.commit()
    db.refresh(journey)

    return journey

@router.delete("/{journey_id}")
def delete_journey(
        journey_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):

    journey = (
        db.query(Journey)
        .filter(
            Journey.id == journey_id,
            Journey.user_id == current_user.id
        )
        .first()
    )

    if not journey:
        raise HTTPException(status_code=404, detail="Journey not found")

    db.delete(journey)
    db.commit()

    return Response(status_code=204)

@router.post("/from-pattern")
def create_journey_from_pattern(
        data: JourneyFromPatternRequest,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):
    memory = (
        db.query(UserMemory)
        .filter(
            UserMemory.id == data.memory_id,
            UserMemory.user_id == current_user.id
        )
        .first()
    )

    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    context = assemble_user_context(
        current_user.id,
        db,
    )

    formatted_context = format_context_for_prompt(
        context,
    )

    generated = generate_journey_challenges(
        title=f"Working on {memory.tag.replace('_', ' ')}",
        description=memory.content,
        user_context=formatted_context,
    )

    journey = Journey(
        user_id=current_user.id,
        title=f"Working on {memory.tag.replace('_', ' ')}",
        description=memory.content,
        source="ai_suggested",
        status="active",
        day_current=1,
        day_total=generated["day_total"],
    )

    db.add(journey)
    db.commit()
    db.refresh(journey)

    for challenge in generated["challenges"]:
        db_challenge = JourneyChallenge(
            journey_id=journey.id,
            day_number=challenge["day_number"],
            title=challenge["title"],
            description=challenge["description"],
            status="pending",
        )
        db.add(db_challenge)

    db.commit()
    db.refresh(journey)

    return {
        "journey": journey,
        "challenges": journey.challenges,
        "today_challenge": journey.challenges[0],
    }
