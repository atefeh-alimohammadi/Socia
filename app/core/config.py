
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str

    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    OLLAMA_URL: str = "http://127.0.0.1:11434"
    LLM_MODEL: str = "qwen2.5:7b"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    CORS_ORIGIN: str = "http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()

