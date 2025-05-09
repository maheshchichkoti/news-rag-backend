# app/main.py
import logging
import sys
import os 
import json # For SSE data formatting if needed
import time
from typing import Annotated # For Header dependency
import traceback # For logging exceptions

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.models import schemas # Import your Pydantic models
from app.services import rag_service, ingestion_service # Import your services
from app.utils import helpers # For session ID generation

# --- Logging Configuration ---
# Configure root logger - This should be done once, preferably at the very start.
logging.basicConfig(
    level=settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else logging.INFO, # Default to INFO if not set
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
        # You could add a FileHandler here if needed:
        # logging.FileHandler("app.log")
    ]
)

# Get a logger for this specific module
logger = logging.getLogger(__name__)

# Optionally, set specific log levels for noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING) # uvicorn access logs can be noisy
logging.getLogger("qdrant_client").setLevel(logging.INFO) # Qdrant can be a bit verbose on DEBUG

# --- FastAPI Application ---
app = FastAPI(
    title="News RAG Chatbot API",
    version="0.1.0",
    description="A RAG-powered chatbot for querying news articles.",
    # docs_url="/api/docs",  # Customize docs URL
    # redoc_url="/api/redoc" # Customize ReDoc URL
)

# --- CORS Middleware ---
# Ensure your frontend origins are correctly listed, especially the deployed one.
# For development, "http://localhost:3000" and "http://localhost:5173" are common.
# If settings.FRONTEND_URL is defined, add it.
origins = settings.CORS_ORIGINS.split(",") if hasattr(settings, 'CORS_ORIGINS') and settings.CORS_ORIGINS else [
    "http://localhost:3000",
    "http://localhost:5173",
]
if hasattr(settings, 'FRONTEND_URL') and settings.FRONTEND_URL:
    if settings.FRONTEND_URL not in origins:
        origins.append(settings.FRONTEND_URL)

logger.info(f"CORS enabled for origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all standard methods
    allow_headers=["*"], # Allows all headers, including X-Session-Id and Content-Type
)

# --- Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    # Add startup delay from environment variable
    startup_delay = int(os.getenv("STARTUP_DELAY", "15"))  # Default to 15 seconds if not set
    logger.info(f"Delaying startup for {startup_delay} seconds to allow services to initialize...")
    time.sleep(startup_delay)
    
    logger.info("FastAPI application startup event commencing...")

    # Critical services initialization checks
    # Qdrant client and embedding model (from ingestion_service)
    if ingestion_service.qdrant_client and ingestion_service.embedding_model and ingestion_service.tokenizer:
        logger.info("ingestion_service components (Qdrant, Embedder, Tokenizer) seem available.")
        logger.info("Ensuring Qdrant collection is ready...")
        if not ingestion_service.ensure_qdrant_collection():
            logger.critical("CRITICAL: Failed to ensure Qdrant collection on startup. This may impact ingestion and RAG.")
            # Depending on severity, you might raise an error to stop startup:
            # raise RuntimeError("Failed to initialize Qdrant collection.")
        else:
            logger.info("Qdrant collection confirmed or created.")
    else:
        missing_ingestion_components = []
        if not ingestion_service.qdrant_client: missing_ingestion_components.append("Qdrant client")
        if not ingestion_service.embedding_model: missing_ingestion_components.append("Embedding model")
        if not ingestion_service.tokenizer: missing_ingestion_components.append("Tokenizer")
        logger.critical(f"CRITICAL: Ingestion service components missing: {', '.join(missing_ingestion_components)}. Application might not function correctly.")

    # RAG service components (Redis, Gemini LLM)
    if not rag_service.redis_client:
        logger.critical("CRITICAL: RAG service Redis client not initialized. Chat history will not work.")
    else:
        logger.info("RAG service Redis client seems available.")

    if not rag_service.generative_llm:
        logger.critical("CRITICAL: RAG service Gemini LLM not initialized. Chat responses will not be generated.")
    else:
        logger.info("RAG service Gemini LLM seems available.")
    
    if not rag_service.query_embedding_model: # This should be same as ingestion_service.embedding_model
        logger.critical("CRITICAL: RAG service query_embedding_model not available.")
    else:
        logger.info("RAG service query_embedding_model seems available.")


    logger.info("FastAPI application startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutdown event commencing...")
    if rag_service.redis_client:
        try:
            await rag_service.redis_client.close()
            logger.info("RAG service Redis client closed successfully.")
        except Exception as e:
            logger.error(f"Error closing RAG service Redis client: {e}", exc_info=True)
    logger.info("FastAPI application shutdown complete.")


# --- API Endpoints ---
@app.get("/health", tags=["Health"], response_model=schemas.HealthResponse)
async def health_check():
    """
    Enhanced health check with memory monitoring
    """
    import psutil
    memory = psutil.virtual_memory()
    
    # Basic checks
    api_status = "ok"
    messages = [
        f"API is responsive (Memory: {memory.percent}% used, {memory.available/1024/1024:.1f}MB available)",
        f"Load: {os.getloadavg()[0]:.2f}"
    ]
    
    # Service checks
    if not all([ingestion_service.qdrant_client, ingestion_service.embedding_model, ingestion_service.tokenizer]):
        api_status = "error"
        messages.append("Ingestion service core components not ready")
    
    if not all([rag_service.redis_client, rag_service.generative_llm, rag_service.query_embedding_model]):
        api_status = "error"
        messages.append("RAG service core components not ready")

    return {
        "status": api_status,
        "message": "; ".join(messages),
        "memory_used_percent": memory.percent,
        "memory_available_mb": memory.available/1024/1024,
        "system_load": os.getloadavg()[0]
    }

# --- Session Management Endpoints ---
@app.post("/session/new", response_model=schemas.SessionResponse, tags=["Session"])
async def create_new_session():
    """Generates and returns a new unique session ID."""
    session_id = helpers.generate_session_id()
    logger.info(f"New session created: {session_id}")
    return schemas.SessionResponse(session_id=session_id, message="New session created.")

# --- Ingestion Endpoint ---
@app.post("/ingest", response_model=schemas.IngestResponse, tags=["Ingestion"], status_code=202) # Default to 202 Accepted
async def ingest_article_endpoint(article_data: schemas.ArticleIngestRequest):
    """
    Ingests a single news article either from a URL or direct text content.
    The process is asynchronous; this endpoint acknowledges the request.
    """
    logger.info(f"Received ingest request - URL: {article_data.url if article_data.url else 'N/A'}, Text provided: {bool(article_data.text_content)}, Source: {article_data.source_name}")
    if not all([ingestion_service.qdrant_client, ingestion_service.embedding_model, ingestion_service.tokenizer]):
        logger.error("Ingestion endpoint called but ingestion service core components are not ready.")
        raise HTTPException(status_code=503, detail="Ingestion service is not ready (core dependencies missing).")

    result = await ingestion_service.ingest_single_article(
        url=article_data.url,
        text_content=article_data.text_content,
        source_name=article_data.source_name
    )
    if result["status"] == "error":
        logger.error(f"Ingestion failed for source '{article_data.source_name}': {result['message']}")
        raise HTTPException(status_code=400, detail=result["message"]) # Bad request if input data led to error
    elif result["status"] == "warning":
        logger.warning(f"Ingestion for source '{article_data.source_name}' completed with warning: {result['message']}")
        # Still return 202, but the message indicates a warning
        return JSONResponse(status_code=202, content=result)
    
    logger.info(f"Ingestion successful for source '{article_data.source_name}', article_id: {result.get('article_id')}")
    return schemas.IngestResponse(**result)


# --- Chat Endpoints ---
@app.post("/chat", tags=["Chat"])
async def stream_chat_response(
    chat_request: schemas.ChatRequest,
    x_session_id: Annotated[str | None, Header(convert_underscores=True)] = None
):
    """
    Handles a user's chat query, performs RAG, and streams Gemini's response.
    Requires `X-Session-Id` header.
    """
    if not x_session_id:
        logger.warning("Chat request received without X-Session-Id header.")
        raise HTTPException(status_code=400, detail="X-Session-Id header is required.")
    
    logger.info(f"Chat request for session_id: {x_session_id}, query: '{chat_request.query[:100]}...'") # Log only snippet

    # Check if all necessary RAG components are up
    critical_rag_components = [
        rag_service.generative_llm,
        rag_service.query_embedding_model,
        rag_service.qdrant_client, # Should be same as ingestion_service.qdrant_client
        rag_service.redis_client
    ]
    if not all(critical_rag_components):
        logger.error(f"Chat service unavailable for session {x_session_id} due to missing critical RAG components.")
        async def error_stream_service_unavailable():
            error_message = "Sorry, the chat service is temporarily unavailable due to internal setup issues. Please try again later."
            yield f"data: {json.dumps({'error': error_message, 'final': True, 'type': 'error'})}\n\n"
        return StreamingResponse(error_stream_service_unavailable(), media_type="text/event-stream")

    async def event_generator():
        full_response_for_log = [] # Use a list to join later, more efficient for many small strings
        try:
            async for content_chunk in rag_service.process_chat_query(chat_request.query, x_session_id):
                full_response_for_log.append(content_chunk)
                yield f"data: {json.dumps({'text': content_chunk, 'type': 'content'})}\n\n"
            # After the stream is finished, send a final "done" event
            yield f"data: {json.dumps({'event': 'done', 'type': 'event'})}\n\n"
            final_response_str = "".join(full_response_for_log)
            logger.info(f"Stream finished for session {x_session_id}. Full response length: {len(final_response_str)}")
        except Exception as e:
            logger.error(f"Error during chat streaming for session {x_session_id}: {e}", exc_info=True)
            error_message = "An unexpected error occurred while processing your request."
            yield f"data: {json.dumps({'error': error_message, 'final': True, 'type': 'error'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/chat/history/{session_id}", response_model=schemas.SessionHistoryResponse, tags=["Chat"])
async def get_session_history(session_id: str):
    """Retrieves the chat history for a given session ID."""
    logger.debug(f"Request for chat history for session_id: {session_id}")
    if not rag_service.redis_client:
        logger.error(f"Attempt to get history for session {session_id} but Redis client unavailable.")
        raise HTTPException(status_code=503, detail="Chat history service (Redis) unavailable.")
    
    history = await rag_service.get_chat_history(session_id)
    logger.info(f"Retrieved {len(history)} messages for session_id: {session_id}")
    return schemas.SessionHistoryResponse(session_id=session_id, history=history)


@app.post("/chat/session/{session_id}/clear", response_model=schemas.ClearSessionResponse, tags=["Chat"])
async def clear_session_history(session_id: str):
    """Clears the chat history for a given session ID."""
    logger.info(f"Request to clear chat history for session_id: {session_id}")
    if not rag_service.redis_client:
        logger.error(f"Attempt to clear history for session {session_id} but Redis client unavailable.")
        raise HTTPException(status_code=503, detail="Chat history service (Redis) unavailable.")
        
    await rag_service.clear_chat_history(session_id)
    logger.info(f"Chat history cleared for session_id: {session_id}")
    return schemas.ClearSessionResponse(session_id=session_id, message="Chat history cleared successfully.")

# --- Main Execution ---
if __name__ == "__main__":
    # This configuration is for running with `python app/main.py`
    # For production, prefer `gunicorn` or `uvicorn` run directly, e.g.:
    # uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
    # The log_level here will be overridden by the basicConfig if not using uvicorn's --log-level
    logger.info("Starting Uvicorn server directly for development...")
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST if hasattr(settings, 'API_HOST') else "0.0.0.0",
        port=settings.API_PORT if hasattr(settings, 'API_PORT') else 8000,
        reload=settings.API_RELOAD if hasattr(settings, 'API_RELOAD') else True,
        log_level=(settings.LOG_LEVEL.lower() if hasattr(settings, 'LOG_LEVEL') else "info")
    )