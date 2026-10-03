from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, Body
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
    ChallengeCompleteRequest,
    ChallengeSkipRequest,
)

from app.utils.token import get_current_user

from app.services.context_assembler import (
    assemble_user_context,
    format_context_for_prompt,
)

from app.services.journey_service import (
    generate_journey_outline,
    generate_next_challenge,
)


router = APIRouter(
    prefix="/journeys",
    tags=["Journeys"],
)


def _create_journey_with_first_challenge(
    db: Session,
    current_user: User,
    title: str,
    description: str,
    source: str,
    source_memory_id: int | None = None,
) -> Journey:

    context = assemble_user_context(
        current_user.id,
        db
    )

    formatted_context = format_context_for_prompt(context)

    outline = generate_journey_outline(
        title=title,
        description=description,
        user_context=formatted_context,
    )

    journey = Journey(
        user_id=current_user.id,
        title=title,
        description=description,
        source=source,
        status="active",
        day_current=1,
        day_total=outline["day_total"],
        source_memory_id=source_memory_id,
    )

    try:
        db.add(journey)

        # Send the INSERT to the database so journey.id is available,
        # but do not commit yet.
        db.flush()

        first_challenge = generate_next_challenge(
            title=title,
            description=description,
            user_context=formatted_context,
            day_number=1,
            previous_challenge=None,
        )

        db_challenge = JourneyChallenge(
            journey_id=journey.id,
            day_number=1,
            title=first_challenge["title"],
            description=first_challenge["description"],
            status="pending",
        )

        db.add(db_challenge)

        # Journey and its first challenge are committed together.
        db.commit()
        db.refresh(journey)

    except Exception:
        db.rollback()
        raise

    return journey


@router.post("/", response_model=JourneyDetailResponse)
def create_journey(
    data: JourneyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    journey = _create_journey_with_first_challenge(
        db=db,
        current_user=current_user,
        title=data.title,
        description=data.description,
        source="user_created",
    )

    return {
        "journey": journey,
        "challenges": journey.challenges,
        "today_challenge": journey.challenges[0],
    }


# =========================================================
# Get Journeys
#
# If status is provided:
#     /journeys/?status=active
#     /journeys/?status=paused
#     /journeys/?status=completed
#
# If status is omitted or empty:
#     /journeys/
#     /journeys/?status=
#
# then ALL journeys are returned, including completed ones.
# =========================================================

@router.get("/", response_model=list[JourneyResponse])
def read_journeys(
    status: Optional[str] = None,
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
        .filter(
            Journey.id == journey_id,
            Journey.user_id == current_user.id,
        )
        .first()
    )

    if not journey:
        raise HTTPException(
            status_code=404,
            detail="Journey not found",
        )

    today = None

    for challenge in journey.challenges:
        if challenge.day_number == journey.day_current:
            today = challenge
            break

    return {
        "journey": journey,
        "challenges": journey.challenges,
        "today_challenge": today,
    }


def _advance_journey_and_generate_next(
    db: Session,
    journey: Journey,
    current_challenge: JourneyChallenge | None,
    current_user: User,
) -> JourneyChallenge | None:
    """
    Generates the next challenge before committing the day transition.

    If challenge generation fails, the transaction is rolled back so
    the journey does not advance to a day without a challenge.

    Returns the newly created challenge, or None if the journey
    is now complete.
    """

    if journey.day_current >= journey.day_total:
        journey.status = "completed"

        db.commit()
        db.refresh(journey)

        return None

    next_day = journey.day_current + 1

    try:
        context = assemble_user_context(
            current_user.id,
            db
        )

        formatted_context = format_context_for_prompt(context)

        next_challenge_data = generate_next_challenge(
            title=journey.title,
            description=journey.description,
            user_context=formatted_context,
            day_number=next_day,
            previous_challenge=current_challenge,
        )

        next_challenge = JourneyChallenge(
            journey_id=journey.id,
            day_number=next_day,
            title=next_challenge_data["title"],
            description=next_challenge_data["description"],
            status="pending",
        )

        # Update the current day only after the next challenge
        # has been successfully generated.
        journey.day_current = next_day

        db.add(next_challenge)

        # Commit the day transition and the new challenge together.
        db.commit()
        db.refresh(journey)

    except Exception:
        db.rollback()
        raise

    return next_challenge


@router.post(
    "/{journey_id}/complete-challenge",
    response_model=JourneyDetailResponse,
)
def complete_challenge(
    journey_id: int,
    feedback: ChallengeCompleteRequest = Body(
        default=ChallengeCompleteRequest()
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    journey = (
        db.query(Journey)
        .filter(
            Journey.id == journey_id,
            Journey.user_id == current_user.id,
        )
        .first()
    )

    if not journey:
        raise HTTPException(
            status_code=404,
            detail="Journey not found",
        )

    if journey.status != "active":
        raise HTTPException(
            status_code=400,
            detail="Journey is not active",
        )

    current_challenge = (
        db.query(JourneyChallenge)
        .filter(
            JourneyChallenge.journey_id == journey_id,
            JourneyChallenge.day_number == journey.day_current,
        )
        .first()
    )

    if not current_challenge:
        raise HTTPException(
            status_code=404,
            detail="Current challenge not found",
        )

    if current_challenge.status != "pending":
        raise HTTPException(
            status_code=400,
            detail="Current challenge has already been completed or skipped",
        )

    current_challenge.status = "completed"
    current_challenge.completed_at = datetime.now(timezone.utc)
    current_challenge.difficulty_feedback = (
        feedback.difficulty_feedback
    )
    current_challenge.emotional_response = (
        feedback.emotional_response
    )

    _advance_journey_and_generate_next(
        db=db,
        journey=journey,
        current_challenge=current_challenge,
        current_user=current_user,
    )

    today = (
        db.query(JourneyChallenge)
        .filter(
            JourneyChallenge.journey_id == journey_id,
            JourneyChallenge.day_number == journey.day_current,
        )
        .first()
    )

    return {
        "journey": journey,
        "challenges": sorted(
            journey.challenges,
            key=lambda x: x.day_number,
        ),
        "today_challenge": today,
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
            Journey.user_id == current_user.id,
        )
        .first()
    )

    if not journey:
        raise HTTPException(
            status_code=404,
            detail="Journey not found",
        )

    if status not in [
        "active",
        "paused",
        "completed",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Invalid status",
        )

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
            Journey.user_id == current_user.id,
        )
        .first()
    )

    if not journey:
        raise HTTPException(
            status_code=404,
            detail="Journey not found",
        )

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
            UserMemory.user_id == current_user.id,
        )
        .first()
    )

    if not memory:
        raise HTTPException(
            status_code=404,
            detail="Memory not found",
        )

    journey = _create_journey_with_first_challenge(
        db=db,
        current_user=current_user,
        title=f"Working on {memory.tag.replace('_', ' ')}",
        description=memory.content,
        source="ai_suggested",
        source_memory_id=memory.id,
    )

    return {
        "journey": journey,
        "challenges": journey.challenges,
        "today_challenge": journey.challenges[0],
    }


@router.post(
    "/{journey_id}/skip-challenge",
    response_model=JourneyDetailResponse,
)
def skip_challenge(
    journey_id: int,
    skip_data: ChallengeSkipRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    journey = (
        db.query(Journey)
        .filter(
            Journey.id == journey_id,
            Journey.user_id == current_user.id,
        )
        .first()
    )

    if not journey:
        raise HTTPException(
            status_code=404,
            detail="Journey not found",
        )

    if journey.status != "active":
        raise HTTPException(
            status_code=400,
            detail="Journey is not active",
        )

    current_challenge = (
        db.query(JourneyChallenge)
        .filter(
            JourneyChallenge.journey_id == journey_id,
            JourneyChallenge.day_number == journey.day_current,
        )
        .first()
    )

    if not current_challenge:
        raise HTTPException(
            status_code=404,
            detail="Current challenge not found",
        )

    if current_challenge.status != "pending":
        raise HTTPException(
            status_code=400,
            detail="Current challenge has already been completed or skipped",
        )

    current_challenge.status = "skipped"
    current_challenge.skip_reason = skip_data.skip_reason
    current_challenge.skip_reason_detail = (
        skip_data.skip_reason_detail
    )

    _advance_journey_and_generate_next(
        db=db,
        journey=journey,
        current_challenge=current_challenge,
        current_user=current_user,
    )

    today = (
        db.query(JourneyChallenge)
        .filter(
            JourneyChallenge.journey_id == journey_id,
            JourneyChallenge.day_number == journey.day_current,
        )
        .first()
    )

    return {
        "journey": journey,
        "challenges": sorted(
            journey.challenges,
            key=lambda x: x.day_number,
        ),
        "today_challenge": today,
    }
