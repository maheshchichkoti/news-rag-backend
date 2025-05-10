# app/core/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List

# Determine DISABLE_RAG status from environment variable
# This needs to be evaluated before the Settings class uses it if you want it as a class variable default
_DISABLE_RAG_ENV_VAR = os.getenv("DISABLE_RAG", "false").lower()
IS_RAG_DISABLED_FROM_ENV = _DISABLE_RAG_ENV_VAR in ("true", "1", "yes")

if IS_RAG_DISABLED_FROM_ENV:
    # Use print as logger might not be configured yet when this module is imported
    print("INFO: [config.py] RAG features are CONFIGURED TO BE DISABLED via DISABLE_RAG environment variable.")
else:
    print("INFO: [config.py] RAG features are CONFIGURED TO BE ENABLED (DISABLE_RAG environment variable not true).")


class Settings(BaseSettings):
    # --- Core API Keys & URLs ---
    # Provide sensible defaults or ensure they are in .env / Render environment
    GOOGLE_API_KEY: str = "YOUR_GOOGLE_API_KEY_IF_NOT_SET_IN_ENV"
    QDRANT_URL: str = "http://localhost:6333" # Will be overridden by Render env var if set
    QDRANT_API_KEY: str | None = None
    REDIS_URL: str = "redis://localhost:6379/0" # Will be overridden by Render env var

    # --- Embedding & Vector DB (not used if RAG disabled) ---
    SENTENCE_TRANSFORMER_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_COLLECTION_NAME: str = "news_articles_v3"

    # --- RAG Parameters (not used if RAG disabled) ---
    CHUNK_SIZE: int = 384
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 3

    # --- LLM (not used if RAG disabled) ---
    GEMINI_MODEL_NAME: str = "models/gemini-1.5-flash-latest"

    # --- Session & Cache Settings (Redis parts always used) ---
    SESSION_ID_HEADER_NAME: str = "X-Session-Id"
    REDIS_SESSION_TTL_SECONDS: int = 3600
    MAX_CHAT_HISTORY_LENGTH: int = 10
    PROMPT_HISTORY_MESSAGES_COUNT: int = 6 # Used if RAG enabled

    # --- Application Behavior ---
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0" # For uvicorn/gunicorn binding
    API_PORT: int = 8000    # For uvicorn/gunicorn binding (matches $PORT in Dockerfile CMD)
    API_RELOAD: bool = False # Should be False for production/docker

    # --- CORS ---
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173" # Default, overridden by env
    FRONTEND_URL: str | None = None # Optional

    # --- Assessment Mode ---
    # This value is derived from the environment variable at module load time
    DISABLE_RAG: bool = IS_RAG_DISABLED_FROM_ENV

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore'
    )

    @property
    def CORS_ORIGINS_LIST(self) -> List[str]:
        # ... (your CORS_ORIGINS_LIST property logic is fine) ...
        if not self.CORS_ORIGINS: return []
        origins = [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
        if self.FRONTEND_URL and self.FRONTEND_URL not in origins:
            origins.append(self.FRONTEND_URL)
        return origins

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()