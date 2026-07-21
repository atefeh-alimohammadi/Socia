from ollama import chat
from fastapi import HTTPException

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
"""

def get_ai_response(user_message: str, conversation_history: list, user_memories: list | None = None) -> str:
    try:

        if user_memories:
            memory_text = "\n".join(
                [f"- {m.content}" for m in user_memories]
            )

            system_content = (
                SYSTEM_PROMPT
                + "\n\nWhat do you know about this user:\n"
                + memory_text
            )
        else:
            system_content = SYSTEM_PROMPT

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
        print("Ollama error:", e)

        raise HTTPException(status_code=502, detail="AI service unavailable")
