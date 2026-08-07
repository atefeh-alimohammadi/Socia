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