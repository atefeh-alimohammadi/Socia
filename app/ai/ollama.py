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

def get_ai_response(user_message: str, conversation_history: list) -> str:
    try:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        for msg in conversation_history:
            role = "assistant" if msg.role == "assistant" else "user"

            messages.append({
                "role": role,
                "content": msg.content
            })

        messages.append({
                "role": "user",
                "content": user_message
            })

        response = chat(
                model="qwen2.5:7b",
                messages=messages,
            )

        return response.message.content

    except Exception as e:
        print("Ollama error:", e)

        raise HTTPException(status_code=502, detail="AI service unavailable")
