import json

from app.ai.ollama import generate_ai_response
import logging

logger = logging.getLogger(__name__)


def generate_journey_challenges(
        title: str,
        description: str,
        user_context: str
) -> dict:

    prompt = f"""
You are Socia's journey generator.

Create a personalized improvement journey for this user.

Journey goal:
{title}

Description:
{description}


What you know about this user:

{user_context}


Generate a structured challenge plan.

Return ONLY valid JSON:

{{
    "day_total": 21,
    "challenges": [
        {{
            "day_number": 1,
            "title": "Short challenge title",
            "description": "Specific actionable challenge written directly to the user"
        }}
    ]
}}


Rules:
- day_total should be between 14 and 30 depending on complexity
- Each challenge must be small and achievable in one day
- Challenges should connect to real social situations
- Avoid generic self-help advice.
- Every challenge must reference at least one user-specific memory if available.
- If no memory exists, create general beginner challenges.
- Use user's context
- Write directly to the user
- Make challenges progressively harder
- Return only JSON
- Do not mention specific apps, websites, brands, or companies unless the user explicitly mentioned them. Keep challenges universally applicable.
- Avoid making assumptions about the user's daily environment. If location or situation is unknown, use generic social situations.
"""

    response = generate_ai_response(prompt)

    try:
        cleaned_response = (
            response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        data = json.loads(cleaned_response)

        if (
            "day_total" not in data
            or "challenges" not in data
            or not isinstance(data["challenges"], list)
        ):
            raise ValueError(
                "Invalid journey format"
            )

        return data

    except Exception as e:

        logger.error(
            "Journey generation failed: %s",
            e,
            exc_info=True
        )
        raise Exception(
            "Journey generation failed. Please try again."
        )


def generate_journey_outline(
        title: str,
        description: str,
        user_context: str
) -> dict:
    """
    Determine a rough target length for a journey without generating
    the full day-by-day challenge plan. Challenges are generated
    progressively, one at a time, via generate_next_challenge.
    """

    prompt = f"""
You are Socia's journey generator.

A user is starting a personalized improvement journey.

Journey goal:
{title}

Description:
{description}

What you know about this user:
{user_context}

Based on the complexity of this goal, estimate a reasonable total
number of days for this journey. This is a rough target, not a fixed
plan - the actual challenges will be created one at a time as the
journey progresses.

Return ONLY valid JSON:

{{
    "day_total": 21
}}

Rules:
- day_total should be between 14 and 30 depending on complexity
- Return only JSON
"""

    response = generate_ai_response(prompt)

    try:
        cleaned_response = (
            response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        data = json.loads(cleaned_response)

        if "day_total" not in data:
            raise ValueError(
                "Invalid outline format"
            )

        day_total = int(data["day_total"])

        if day_total < 14:
            day_total = 14
        elif day_total > 30:
            day_total = 30

        return {"day_total": day_total}

    except Exception as e:

        logger.error(
            "Journey outline generation failed: %s",
            e,
            exc_info=True
        )
        raise Exception(
            "Journey outline generation failed. Please try again."
        )


def _format_previous_outcome(
        previous_challenge,
        previous_outcome: str,
) -> str:

    if previous_challenge is None:
        return "This is the first challenge of the journey."

    lines = [
        f'Previous challenge (day {previous_challenge.day_number}): '
        f'"{previous_challenge.title}" - {previous_challenge.description}',
        f"Outcome: {previous_outcome}",
    ]

    if previous_challenge.status == "completed":
        if previous_challenge.difficulty_feedback:
            lines.append(
                f"User's difficulty feedback: {previous_challenge.difficulty_feedback}"
            )
        if previous_challenge.emotional_response:
            lines.append(
                f"User's emotional response: {previous_challenge.emotional_response}"
            )

    elif previous_challenge.status == "skipped":
        if previous_challenge.skip_reason:
            lines.append(
                f"Reason the user skipped it: {previous_challenge.skip_reason}"
            )
        if previous_challenge.skip_reason_detail:
            lines.append(
                f"Additional detail: {previous_challenge.skip_reason_detail}"
            )

    return "\n".join(lines)


def generate_next_challenge(
        title: str,
        description: str,
        user_context: str,
        day_number: int,
        previous_challenge=None,
) -> dict:
    """
    Generate a single challenge for the given day, taking into account
    the outcome of the previous challenge if one exists.
    """

    if previous_challenge is not None:
        previous_outcome = (
            "completed" if previous_challenge.status == "completed"
            else "skipped"
        )
    else:
        previous_outcome = None

    previous_context = _format_previous_outcome(
        previous_challenge,
        previous_outcome,
    )
    logger.info("Next challenge prompt context:\n%s", previous_context)

    prompt = f"""
You are Socia's journey generator.

This is part of an ongoing personalized improvement journey.

Journey goal:
{title}

Description:
{description}

What you know about this user:
{user_context}

Progress so far:
{previous_context}

Generate the next single challenge for day {day_number} of this journey.

Return ONLY valid JSON:

{{
    "title": "Short challenge title",
    "description": "Specific actionable challenge written directly to the user"
}}

Rules:
- The challenge must be small and achievable in one day.
- The challenge should connect to a real social situation.
- Avoid generic self-help advice.
- Write directly to the user.
- Take the outcome of the previous challenge into account: if it was
  skipped, consider why, and respond appropriately rather than simply
  repeating the same type of challenge. If it was completed, build on
  that progress.
- Do not mention specific apps, websites, brands, or companies unless
  the user explicitly mentioned them.
- Avoid making assumptions about the user's daily environment. If
  location or situation is unknown, use generic social situations.
- Return only JSON.
"""

    response = generate_ai_response(prompt)

    try:
        cleaned_response = (
            response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        data = json.loads(cleaned_response)

        if "title" not in data or "description" not in data:
            raise ValueError(
                "Invalid challenge format"
            )

        return data

    except Exception as e:

        logger.error(
            "Next challenge generation failed: %s",
            e,
            exc_info=True
        )
        raise Exception(
            "Challenge generation failed. Please try again."
        )