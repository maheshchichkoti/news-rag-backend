# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List
import os # For getenv

class Settings(BaseSettings):
    # --- Core API Keys & URLs ---
    GOOGLE_API_KEY: str = "YOUR_GOOGLE_API_KEY_HERE_IF_NOT_IN_ENV" # Provide a default or ensure it's in .env
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
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
    REDIS_SESSION_TTL_SECONDS: int = 3600
    MAX_CHAT_HISTORY_LENGTH: int = 10
    PROMPT_HISTORY_MESSAGES_COUNT: int = 6

    # --- Application Behavior ---
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000 # This is the internal port Gunicorn/Uvicorn will listen on
    API_RELOAD: bool = True # For local Uvicorn reload

    # --- CORS ---
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    FRONTEND_URL: str | None = None

    # --- Assessment Mode ---
    # This will be read from environment variable DISABLE_RAG
    # Pydantic-settings automatically tries to cast env vars to bool
    DISABLE_RAG: bool = False # Default to RAG enabled

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
    # For Render, DISABLE_RAG should be set as an environment variable to "1" or "true"
    # This ensures pydantic-settings correctly interprets it.
    disable_rag_env = os.getenv("DISABLE_RAG", "false").lower()
    effective_disable_rag = disable_rag_env in ("true", "1", "yes")
    
    # Create settings instance, potentially overriding DISABLE_RAG based on environment
    current_settings = Settings()
    if os.getenv("DISABLE_RAG") is not None: # If env var is set
        current_settings.DISABLE_RAG = effective_disable_rag
        
    if current_settings.DISABLE_RAG:
        print("INFO: RAG features are DISABLED via configuration.") # Use print here as logger might not be set up
    else:
        print("INFO: RAG features are ENABLED via configuration.")
    return current_settings

settings = get_settings()