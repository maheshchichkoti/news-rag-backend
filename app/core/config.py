# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    # --- Core API Keys & URLs ---
    GOOGLE_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None # Optional, depending on your Qdrant setup
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Embedding & Vector DB ---
    SENTENCE_TRANSFORMER_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_COLLECTION_NAME: str = "news_articles_v3"

    # --- RAG Parameters ---
    CHUNK_SIZE: int = 384
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 3

    # --- LLM ---
    GEMINI_MODEL_NAME: str = "models/gemini-1.5-flash-latest"

    # --- Session & Cache Settings ---
    SESSION_ID_HEADER_NAME: str = "X-Session-Id"
    REDIS_SESSION_TTL_SECONDS: int = 3600 # TTL for chat history in Redis (e.g., 1 hour)
    MAX_CHAT_HISTORY_LENGTH: int = 10      # Max *pairs* of user/assistant messages for Redis storage trimming
    PROMPT_HISTORY_MESSAGES_COUNT: int = 6 # Max *individual* messages (e.g., 3 pairs) for LLM prompt context

    # --- Application Behavior ---
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # --- CORS ---
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    FRONTEND_URL: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore'
    )

    @property
    def CORS_ORIGINS_LIST(self) -> List[str]:
        if not self.CORS_ORIGINS:
            return []
        origins = [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
        if self.FRONTEND_URL and self.FRONTEND_URL not in origins:
            origins.append(self.FRONTEND_URL)
        return origins

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()