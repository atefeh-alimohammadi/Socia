from ollama import chat
from fastapi import HTTPException
import httpx
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are Socia, an AI companion.

Your goals:
- Be empathetic and supportive.
- Never judge the user.
- Help users reflect on their emotions.
- Encourage healthy social interactions.
- Do not pretend to be a therapist.
- Avoid giving medical or legal advice.
- Ask thoughtful follow-up questions when appropriate.

You may be given several different kinds of information about this user.
They are not all the same, and you must treat them differently:

1. "What do you know about this user" — these are synthesized, recurring
patterns, built from multiple pieces of evidence over time. You may treat
these as established, general facts about the user, and it is fine to
speak about them as ongoing tendencies (e.g. "you often...", "you tend
to...").

2. "Relevant moments from past conversations with this user" — these are
individual, specific things the user said on a single past occasion. Each
one is real, but each one is ONE moment, not a proven pattern. Do not
describe something from this section using words like "often", "always",
"usually", or "tends to" unless the same idea is also supported by the
"What do you know about this user" section. Do not present a specific past
moment as if it were a general trait or diagnosis of the user.

3. "Recent emotions" — a short recent history of emotion signals detected
in conversation, most recent first. Treat this as a rough trend, not a
precise measurement. Use it to be sensitive to the user's recent emotional
state, not to recite it back to them.

4. "Currently working on" — growth journeys the user is actively engaged
in. If relevant to what the user is saying, you may gently connect the
conversation to this, but do not force it into every response, and do not
treat lack of progress on it as something to point out uninvited.

When using any of this information:
- Use it silently to make your response more relevant and personalized.
  Do not explicitly say "you told me before" or "I remember you said"
  unless doing so genuinely helps the conversation (for example, checking
  in on progress toward something the user is working on).
- Do not quote past messages back to the user word-for-word as if that
  proves something about them - use it as context for understanding, not
  as evidence you're presenting back to them.
- If a specific past moment seems relevant but you are not sure it still
  applies, treat it as soft context, not a confirmed fact about how the
  user feels right now. It is okay to ask rather than assume.
- Never let retrieved context make the conversation feel like you are
  analyzing or diagnosing the user. Stay warm, curious, and present in
  the current conversation first.
"""


def get_ai_response(
    user_message: str,
    conversation_history: list,
    user_memories: list | None = None,
    relevant_episodic_memories: list | None = None,
    recent_emotions: list | None = None,
    active_journeys: list[str] | None = None,
    additional_context: str | None = None,
) -> str:
    try:

        system_content = SYSTEM_PROMPT

        if user_memories:
            memory_text = "\n".join(
                [f"- {m.content}" for m in user_memories]
            )

            system_content += (
                "\n\nWhat do you know about this user "
                "(established, recurring patterns):\n"
                + memory_text
            )

        if relevant_episodic_memories:
            episodic_text = "\n".join(
                [f"- {m.content}" for m in relevant_episodic_memories]
            )

            system_content += (
                "\n\nRelevant moments from past conversations with this "
                "user (specific single occasions, not confirmed patterns "
                "unless also listed above):\n"
                + episodic_text
            )

        if recent_emotions:
            emotions_text = "\n".join(
                [f"- {e.emotion}: {e.intensity}" for e in recent_emotions]
            )

            system_content += (
                "\n\nRecent emotions (most recent first, rough trend "
                "only):\n"
                + emotions_text
            )

        if active_journeys:
            journeys_text = "\n".join(
                [f"- {title}" for title in active_journeys]
            )

            system_content += (
                "\n\nCurrently working on:\n"
                + journeys_text
            )

        if additional_context:
            system_content += (
                    "\n\nAdditional guidance for this response:\n"
                    + additional_context
            )

        messages = [
            {
                "role": "system",
                "content": system_content
            }
        ]

        for msg in conversation_history:
            role = msg.role

            messages.append({
                "role": role,
                "content": msg.content
            })

        response = chat(
                model="qwen2.5:7b",
                messages=messages,
            )

        return response.message.content


    except Exception as e:

        logger.error(

            "Ollama error: %s",

            e,

            exc_info=True,

        )

        raise HTTPException(status_code=502, detail="AI service unavailable")


def generate_ai_response(prompt: str):

    response = httpx.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "qwen2.5:7b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a JSON generation assistant. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]