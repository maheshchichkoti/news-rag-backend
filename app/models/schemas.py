# app/models/schemas.py
from pydantic import BaseModel, Field, model_validator # <<< MOVED model_validator HERE
from typing import List, Optional, Dict, Any

# --- Ingestion Related Schemas ---
class ArticleIngestRequest(BaseModel):
    url: str | None = None
    text_content: str | None = None
    source_name: str | None = None

    @model_validator(mode='before')
    @classmethod
    def check_url_or_text_content_present(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get('url') and not data.get('text_content'):
            raise ValueError('Either "url" or "text_content" must be provided')
        return data

class IngestResponse(BaseModel):
    status: str
    message: str
    article_id: Optional[str] = None
    num_chunks: Optional[int] = None

# --- Chat Related Schemas ---
class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ChatRequest(BaseModel): # Query is the only field now
    query: str

class SessionResponse(BaseModel): # For creating a new session
    session_id: str
    message: str

class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[ChatMessage]

class ClearSessionResponse(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel): # Optional, for non-streaming if ever needed
    session_id: str
    answer: str
    retrieved_context: Optional[List[Dict[str, Any]]] = None

class HealthResponse(BaseModel):
    status: str # "ok" or "error"
    message: str