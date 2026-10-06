import logging

from app.ai.ollama import generate_ai_response

logger = logging.getLogger(__name__)


TITLE_GENERATION_PROMPT = """
You are generating a short title for a conversation in an AI companion app.

User's first message:
"{user_message}"

Assistant's response:
"{assistant_response}"

Generate ONE short, natural conversation title that describes the main topic
of the conversation.

Requirements:
- 3 to 7 words.
- Write in natural English.
- Use sentence case.
- Do not start with "Conversation", "Chat", or "Discussion".
- Do not mention Socia.
- Do not diagnose or label the user.
- Do not describe the user as a personality type.
- Focus on the topic or situation being discussed.
- Do not use quotation marks.
- Return ONLY the title.
"""


def generate_conversation_title(
    user_message: str,
    assistant_response: str,
) -> str | None:
    """
    Generate a short title for a conversation.

    Returns None if generation fails or produces unusable output.
    """

    try:
        prompt = TITLE_GENERATION_PROMPT.format(
            user_message=user_message[:1000],
            assistant_response=assistant_response[:1500],
        )

        response = generate_ai_response(prompt)

        title = response.strip()

        # Remove accidental surrounding quotes.
        title = title.strip("\"'")

        # Normalize whitespace.
        title = " ".join(title.split())

        if not title:
            return None

        # Avoid obviously invalid/generated placeholders.
        invalid_titles = {
            "conversation",
            "new conversation",
            "untitled conversation",
            "chat",
            "discussion",
        }

        if title.lower() in invalid_titles:
            return None

        # Keep titles short enough for the UI/database.
        if len(title) > 80:
            title = title[:80].rstrip()

        return title

    except Exception as exc:
        logger.warning(
            "Conversation title generation failed: %s",
            exc,
            exc_info=True,
        )

        return None

