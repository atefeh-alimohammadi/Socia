from google import genai

from fastapi import HTTPException, status

from app.core.config import settings


client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)

SYSTEM_PROMPT = """
You are an AI companion and social coaching assistant.

Your role:
- Be patient, empathetic, and non-judgmental.
- Help users reflect on emotions, communication, and social situations.
- Ask thoughtful questions.
- Encourage real-world growth and healthy relationships.
- Do not diagnose mental health conditions.
"""

def get_ai_response(user_message:str, conversation_history: list) -> str:

    try:
        model = "gemini-2.5-flash"

        history = []

        for message in conversation_history:
            role = "model" if message.role == "assistant" else "user"

            history.append(
                {
                    "role": role,
                    "parts": [
                        {
                            "text": message.content,
                        }
                    ]
                }
            )

            chat = client.chats.create(
                model=model,
                history=history,
                config={
                    "system_instruction": SYSTEM_PROMPT
                }
            )
            response = chat.send_message(user_message)

            return response.text

    except Exception as e:
        print("Gemini error:", e)

        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="AI service unavailable")
