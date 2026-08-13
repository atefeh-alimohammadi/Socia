import json
import logging

from app.ai.ollama import generate_ai_response


logger = logging.getLogger(__name__)


def check_response_consistency(
        response: str,
        user_memories: list,
        recent_emotions: list,
) -> dict:

    memories_text = "\n".join(
        f"- {memory.content}"
        for memory in user_memories
    )

    emotions_text = "\n".join(
        f"- {emotion.emotion}: {emotion.intensity}"
        for emotion in recent_emotions
    )

    prompt = f"""
    You are a response quality checker for an AI companion.

    Check whether the assistant response is consistent with the known
    information about the user.

    Assistant response:
    "{response}"

    Known user memories:
    {memories_text if memories_text else "(none)"}

    Recent emotions:
    {emotions_text if emotions_text else "(none)"}

    Check specifically:
    1. Does the response contradict known user information?
    2. Does it ignore clear recent emotional distress when that distress
       is relevant to the response?
    3. Is there a clear, direct inconsistency that should be corrected?

    Do NOT flag a response as inconsistent merely because it:
    - does not explicitly reference user memories, episodic memories, or
      recent emotions
    - introduces a new topic the user just raised, even if that topic
      isn't connected to anything in the known user information
    - fails to personalize the response

    Lack of personalization is not an inconsistency. Only flag a response
    if it actively contradicts something known about the user, or clearly
    ignores relevant, recent emotional distress that the response should
    have been sensitive to.

    Return ONLY valid JSON:

    {{
        "consistent": true,
        "issue": null
    }}

    or:

    {{
        "consistent": false,
        "issue": "Brief explanation of the inconsistency."
    }}

    Do not invent inconsistencies.
    If there is no clear problem, return consistent=true.
    """
    try:

        raw_response = generate_ai_response(prompt)

        cleaned_response = (
            raw_response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        result = json.loads(cleaned_response)

        if "consistent" not in result:

            raise ValueError(
                "Invalid reflection response format"
            )

        return {
            "consistent": bool(
                result["consistent"]
            ),
            "issue": result.get("issue"),
        }
    except Exception as e:
        logger.error(
            "Reflection check failed: %s",
            e,
            exc_info=True,
        )

        return {
            "consistent": True,
            "issue": None,
        }