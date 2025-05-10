# app/main.py
import logging
import sys
import os
import json
import time
# import psutil # psutil might not be in requirements-minimal.txt, conditional import

from typing import Annotated
import traceback

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings # This will now print RAG status on import
from app.models import schemas
from app.services import rag_service, ingestion_service
from app.utils import helpers

# --- Logging Configuration ---
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
# ... other logger configs

# --- FastAPI Application ---
app = FastAPI(
    title="News RAG Chatbot API",
    version="0.1.0",
    description="A RAG-powered chatbot for querying news articles."
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS enabled for origins: {settings.CORS_ORIGINS_LIST}")


# --- Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    # The STARTUP_DELAY env var is used by Docker CMD.
    # Render has its own "Initial Health Check Delay" setting.
    # The phased startup here is an application-level delay/check.
    logger.info("FastAPI application startup event commencing...")

    if settings.DISABLE_RAG:
        logger.warning("ASSESSMENT MODE: RAG features are disabled. Skipping RAG component checks.")
        # Only check Redis if it's essential for even disabled mode
        if not rag_service.redis_client:
            logger.critical("CRITICAL: RAG service Redis client not initialized. Basic session features may fail.")
        else:
            logger.info("RAG service Redis client seems available (for session management).")
    else:
        logger.info("FULL MODE: Performing RAG component checks...")
        # Your existing checks for ingestion_service and rag_service components
        if ingestion_service.qdrant_client and ingestion_service.embedding_model and ingestion_service.tokenizer:
            logger.info("Ingestion service components (Qdrant, Embedder, Tokenizer) seem available.")
            if not ingestion_service.ensure_qdrant_collection():
                logger.critical("CRITICAL: Failed to ensure Qdrant collection on startup.")
            else:
                logger.info("Qdrant collection confirmed or created.")
        else:
            logger.critical("CRITICAL: One or more Ingestion service components missing.")

        if not rag_service.redis_client: logger.critical("CRITICAL: RAG service Redis client not initialized.")
        else: logger.info("RAG service Redis client seems available.")
        if not rag_service.generative_llm: logger.critical("CRITICAL: RAG service Gemini LLM not initialized.")
        else: logger.info("RAG service Gemini LLM seems available.")
        if not rag_service.query_embedding_model: logger.critical("CRITICAL: RAG service query_embedding_model not available.")
        else: logger.info("RAG service query_embedding_model seems available.")

    logger.info("FastAPI application startup sequence complete.")


@app.on_event("shutdown")
async def shutdown_event():
    # ... (your existing shutdown_event code is fine) ...
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
    api_status = "ok"
    mode_message = "RAG features ENABLED."
    component_messages = []

    # Try to import psutil only if needed and handle if not present in minimal
    try:
        import psutil
        memory = psutil.virtual_memory()
        load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else -1 # getloadavg not on all OS e.g. Windows
        base_message = f"API is responsive (Memory: {memory.percent:.1f}% used, {memory.available/1024/1024:.1f}MB available, Load: {load_avg:.2f})"
    except ImportError:
        base_message = "API is responsive (psutil not installed for detailed memory/load)."
        psutil = None # Ensure psutil is None if import fails

    if settings.DISABLE_RAG:
        mode_message = "RAG features DISABLED (Assessment Mode)."
        if not rag_service.redis_client:
            api_status = "error"
            component_messages.append("Redis client (for sessions) not ready.")
    else: # Full mode checks
        if not all([ingestion_service.qdrant_client, ingestion_service.embedding_model, ingestion_service.tokenizer]):
            api_status = "error" # or "warning"
            component_messages.append("Ingestion service components not ready.")
        if not all([rag_service.redis_client, rag_service.generative_llm, rag_service.query_embedding_model]):
            api_status = "error" # or "warning"
            component_messages.append("RAG service ML components not ready.")
        if not rag_service.redis_client : # Check redis separately if it was missed
             api_status = "error"
             component_messages.append("Redis client not ready.")


    final_message = f"{base_message}; {mode_message}"
    if component_messages:
        final_message += "; " + "; ".join(component_messages)

    response_data = {"status": api_status, "message": final_message}
    if psutil: # Only add memory/load if psutil was available
        response_data["memory_used_percent"] = memory.percent
        response_data["memory_available_mb"] = memory.available/1024/1024
        response_data["system_load"] = load_avg
        
    return schemas.HealthResponse(**response_data) # Use dict unpacking

# ... (your other endpoints: /session/new, /ingest, /chat, /chat/history, /chat/session/{session_id}/clear)
# They will now behave differently based on settings.DISABLE_RAG due to changes in the service layer.

@app.post("/ingest", response_model=schemas.IngestResponse, tags=["Ingestion"], status_code=202)
async def ingest_article_endpoint(article_data: schemas.ArticleIngestRequest):
    if settings.DISABLE_RAG:
        logger.warning("Attempt to use /ingest endpoint while RAG is disabled.")
        # Consider raising HTTPException or returning a specific message
        raise HTTPException(status_code=403, detail="Ingestion is disabled in the current mode.")
        # return JSONResponse(status_code=403, content={"status": "error", "message": "Ingestion is disabled in this mode."})

    # ... (rest of your ingest_article_endpoint logic, it will call the updated ingestion_service)
    logger.info(f"Received ingest request - URL: {article_data.url if article_data.url else 'N/A'}, Text provided: {bool(article_data.text_content)}, Source: {article_data.source_name}")
    if not all([ingestion_service.qdrant_client, ingestion_service.embedding_model, ingestion_service.tokenizer]): # This check might be redundant if service handles it
        logger.error("Ingestion endpoint: Ingestion service core components are not ready.")
        raise HTTPException(status_code=503, detail="Ingestion service is not ready.")

    result = await ingestion_service.ingest_single_article(
        url=article_data.url, text_content=article_data.text_content, source_name=article_data.source_name
    )
    if result["status"] == "error":
        # Log specific error from service if available
        logger.error(f"Ingestion failed for source '{article_data.source_name}': {result.get('message', 'Unknown error')}")
        # Map service error to HTTP error
        status_code = 400 if "content" in result.get('message','').lower() else 500 # Example logic
        raise HTTPException(status_code=status_code, detail=result["message"])
    elif result["status"] == "warning":
        logger.warning(f"Ingestion for source '{article_data.source_name}' warning: {result['message']}")
        return JSONResponse(status_code=202, content=result)
    
    logger.info(f"Ingestion successful for source '{article_data.source_name}', article_id: {result.get('article_id')}")
    return schemas.IngestResponse(**result)


@app.post("/chat", tags=["Chat"])
async def stream_chat_response(
    chat_request: schemas.ChatRequest,
    x_session_id: Annotated[str | None, Header(convert_underscores=True)] = None
):
    if not x_session_id:
        logger.warning("Chat request without X-Session-Id header.")
        raise HTTPException(status_code=400, detail="X-Session-Id header is required.")
    
    logger.info(f"Chat request for session_id: {x_session_id}, query: '{chat_request.query[:100]}...'")

    # Service availability check (covers DISABLE_RAG implicitly via component status)
    # Redis is always checked as it's needed for sessions
    if not rag_service.redis_client:
         logger.error(f"Chat service unavailable for session {x_session_id}: Redis client missing.")
         async def error_stream_redis():
            error_message = "Chat service is temporarily unavailable (session error)."
            yield f"data: {json.dumps({'error': error_message, 'final': True, 'type': 'error'})}\n\n"
         return StreamingResponse(error_stream_redis(), media_type="text/event-stream")

    if not settings.DISABLE_RAG: # Only check these if RAG is supposed to be enabled
        critical_rag_components = [
            rag_service.generative_llm,
            rag_service.query_embedding_model,
            # rag_service.qdrant_client_rag, # Already checked by query_embedding_model if it comes from ingestion
        ]
        if not all(critical_rag_components) or not rag_service.qdrant_client_rag: # Explicitly check qdrant_client_rag too
            logger.error(f"Chat service unavailable for session {x_session_id} due to missing RAG ML components.")
            async def error_stream_ml_unavailable():
                error_message = "AI features are temporarily unavailable. Please try again later."
                yield f"data: {json.dumps({'error': error_message, 'final': True, 'type': 'error'})}\n\n"
            return StreamingResponse(error_stream_ml_unavailable(), media_type="text/event-stream")

    async def event_generator():
        # ... (your event_generator logic is mostly fine, it calls rag_service.process_chat_query)
        # process_chat_query will handle DISABLE_RAG internally
        full_response_for_log = []
        try:
            async for content_chunk in rag_service.process_chat_query(chat_request.query, x_session_id):
                full_response_for_log.append(content_chunk)
                yield f"data: {json.dumps({'text': content_chunk, 'type': 'content'})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'type': 'event'})}\n\n"
            logger.info(f"Stream finished for session {x_session_id}. Full response logged (len): {len(''.join(full_response_for_log))}")
        except Exception as e:
            logger.error(f"Error during chat streaming for session {x_session_id}: {e}", exc_info=True)
            error_message = "An unexpected error occurred while processing your request."
            yield f"data: {json.dumps({'error': error_message, 'final': True, 'type': 'error'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ... (get_session_history, clear_session_history are fine as they only use Redis)
@app.get("/chat/history/{session_id}", response_model=schemas.SessionHistoryResponse, tags=["Chat"])
async def get_session_history(session_id: str):
    logger.debug(f"Request for chat history for session_id: {session_id}")
    if not rag_service.redis_client:
        logger.error(f"Attempt to get history for session {session_id} but Redis client unavailable.")
        raise HTTPException(status_code=503, detail="Chat history service (Redis) unavailable.")
    history = await rag_service.get_chat_history(session_id)
    logger.info(f"Retrieved {len(history)} messages for session_id: {session_id}")
    if not history: # Optional: Be more explicit for empty history on a known session
        # Could check if session key actually exists in Redis vs truly empty
        # For now, returning empty list is fine as per Pydantic schema
        logger.info(f"No history found for session_id: {session_id} (or session does not exist).")
    return schemas.SessionHistoryResponse(session_id=session_id, history=history)

@app.post("/chat/session/{session_id}/clear", response_model=schemas.ClearSessionResponse, tags=["Chat"])
async def clear_session_history(session_id: str):
    logger.info(f"Request to clear chat history for session_id: {session_id}")
    if not rag_service.redis_client:
        logger.error(f"Attempt to clear history for session {session_id} but Redis client unavailable.")
        raise HTTPException(status_code=503, detail="Chat history service (Redis) unavailable.")
    await rag_service.clear_chat_history(session_id) # This service function already logs
    return schemas.ClearSessionResponse(session_id=session_id, message="Chat history cleared successfully.")


# --- Main Execution ---
if __name__ == "__main__":
    # ... (your existing __main__ block is fine) ...
    logger.info("Starting Uvicorn server directly for development...")
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=(settings.LOG_LEVEL.lower())
    )