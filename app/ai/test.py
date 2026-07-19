from google import genai

from app.core.config import settings


client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)
#
#
# for model in client.models.list():
#     print(model.name)

response = client.models.generate_content(
    model="gemini-2.0-flash-lite",
    contents="Hello"
)

print(response.text)