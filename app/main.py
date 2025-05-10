# app/main.py
import logging
import sys
import os
import json
# import psutil # Only import if available and needed for health, not in requirements-minimal
from typing import Annotated
import traceback

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings # This now prints RAG status based on env var
from app.models import schemas
from app.services import rag_service, ingestion_service # These will init based on settings.DISABLE_RAG
from app.utils import helpers

# --- Logging Configuration ---
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# logging.getLogger("uvicorn").setLevel(logging.WARNING) # uvicorn.access can be noisy
# logging.getLogger("gunicorn").setLevel(logging.WARNING)

app = FastAPI(
    title="News RAG Chatbot API",
    version="0.1.0",
    description="A RAG-powered chatbot for querying news articles."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST, # From config property
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS enabled for origins: {settings.CORS_ORIGINS_LIST}")
logger.info(f"Application starting in RAG DISABLED mode: {settings.DISABLE_RAG}")


@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup commencing...")
    # Client initializations are now handled within each service module based on settings.DISABLE_RAG
    # This startup event can primarily log the status determined by those initializations.
    if settings.DISABLE_RAG:
        logger.warning("STARTUP: RAG features are DISABLED. Only core API and session management will be active.")
        if not rag_service.redis_client: # Still check Redis as it's essential
            logger.critical("STARTUP CRITICAL: Redis client in RAG service FAILED to initialize.")
        else:
            logger.info("STARTUP: Redis client (for sessions) in RAG service is available.")
    else:
        logger.info("STARTUP: RAG features are ENABLED. Verifying components...")
        # Ingestion service components
        if not all([ingestion_service.qdrant_client, ingestion_service.embedding_model, ingestion_service.tokenizer, ingestion_service.trafilatura_extract_func]):
            logger.critical("STARTUP CRITICAL: One or more INGESTION RAG components FAILED to initialize.")
        else:
            logger.info("STARTUP: Ingestion RAG components seem initialized.")
            if not ingestion_service.ensure_qdrant_collection(): # This logs its own errors
                logger.critical("STARTUP CRITICAL: Qdrant collection check/setup FAILED.")
            else:
                logger.info("STARTUP: Qdrant collection is ready.")
        # RAG service components
        if not rag_service.redis_client: logger.critical("STARTUP CRITICAL: RAG service Redis client FAILED.")
        if not all([rag_service.query_embedding_model, rag_service.qdrant_client_rag, rag_service.generative_llm]):
            logger.critical("STARTUP CRITICAL: One or more RAG service ML/AI components FAILED to initialize.")
        else:
            logger.info("STARTUP: RAG service ML/AI components seem initialized.")
    logger.info("FastAPI application startup sequence complete.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutdown commencing...")
    if rag_service.redis_client:
        try:
            await rag_service.redis_client.close()
            logger.info("Redis client closed.")
        except Exception as e:
            logger.error(f"Error closing Redis client: {e}", exc_info=True)
    logger.info("FastAPI application shutdown complete.")


@app.get("/health", tags=["Health"], response_model=schemas.HealthResponse)
async def health_check():
    api_status = "ok"
    status_messages = ["API is responsive."]

    if settings.DISABLE_RAG:
        status_messages.append("RAG features: DISABLED (Assessment Mode).")
        if not rag_service.redis_client:
            api_status = "error"
            status_messages.append("Critical: Redis client (for sessions) is NOT ready.")
        else:
            status_messages.append("Redis client: OK.")
    else: # Full RAG mode checks
        status_messages.append("RAG features: ENABLED.")
        ready_ingestion = all([
            ingestion_service.qdrant_client, ingestion_service.embedding_model,
            ingestion_service.tokenizer, ingestion_service.trafilatura_extract_func
        ])
        ready_rag_ml = all([
            rag_service.query_embedding_model, rag_service.qdrant_client_rag,
            rag_service.generative_llm
        ])
        ready_redis = bool(rag_service.redis_client)

        if not ready_ingestion: status_messages.append("Ingestion RAG components: NOT ready.")
        else: status_messages.append("Ingestion RAG components: OK.")
        if not ready_rag_ml: status_messages.append("Chat RAG ML components: NOT ready.")
        else: status_messages.append("Chat RAG ML components: OK.")
        if not ready_redis: status_messages.append("Redis client: NOT ready.")
        else: status_messages.append("Redis client: OK.")

        if not (ready_ingestion and ready_rag_ml and ready_redis):
            api_status = "error" # Or "warning" if some partial functionality is acceptable

    # psutil for memory (optional, attempt import)
    try:
        import psutil
        memory = psutil.virtual_memory()
        load_avg_str = f"{os.getloadavg()[0]:.2f}" if hasattr(os, 'getloadavg') else "N/A"
        status_messages.append(f"Memory: {memory.percent:.1f}% used ({memory.available/1024/1024:.1f}MB free). Load: {load_avg_str}.")
    except ImportError:
        status_messages.append("Memory/Load: (psutil not installed in minimal mode).")
    except Exception as e_psutil:
         logger.warning(f"Could not get psutil info: {e_psutil}")
         status_messages.append("Memory/Load: Error retrieving system stats.")


    return schemas.HealthResponse(status=api_status, message=" | ".join(status_messages))

# --- Session Management Endpoints (should always work) ---
@app.post("/session/new", response_model=schemas.SessionResponse, tags=["Session"])
async def create_new_session():
    # ... (your existing code is fine)
    session_id = helpers.generate_session_id()
    logger.info(f"New session created: {session_id}")
    return schemas.SessionResponse(session_id=session_id, message="New session created.")

@app.get("/chat/history/{session_id}", response_model=schemas.SessionHistoryResponse, tags=["Chat"])
async def get_session_history(session_id: str):
    # ... (your existing code is fine, relies on rag_service.redis_client)
    if not rag_service.redis_client:
        raise HTTPException(status_code=503, detail="Session service unavailable (Redis).")
    history = await rag_service.get_chat_history(session_id)
    return schemas.SessionHistoryResponse(session_id=session_id, history=history)

@app.post("/chat/session/{session_id}/clear", response_model=schemas.ClearSessionResponse, tags=["Chat"])
async def clear_session_history(session_id: str):
    # ... (your existing code is fine, relies on rag_service.redis_client)
    if not rag_service.redis_client:
        raise HTTPException(status_code=503, detail="Session service unavailable (Redis).")
    await rag_service.clear_chat_history(session_id)
    return schemas.ClearSessionResponse(session_id=session_id, message="Chat history cleared successfully.")

# --- Ingestion Endpoint (conditionally active) ---
@app.post("/ingest", response_model=schemas.IngestResponse, tags=["Ingestion"], status_code=202)
async def ingest_article_endpoint(article_data: schemas.ArticleIngestRequest):
    if settings.DISABLE_RAG:
        logger.warning("Attempt to use /ingest endpoint while RAG is disabled.")
        raise HTTPException(status_code=403, detail="Ingestion is disabled in assessment mode.")
    
    # Check if ingestion components are actually ready (they should be if RAG not disabled)
    if not all([ingestion_service.qdrant_client, ingestion_service.embedding_model, ingestion_service.tokenizer, ingestion_service.trafilatura_extract_func]):
         logger.error("Ingestion endpoint called, but RAG components in ingestion_service are not ready.")
         raise HTTPException(status_code=503, detail="Ingestion service components are not available.")

    # ... (rest of your existing ingest_article_endpoint logic is fine)
    logger.info(f"Received ingest request - URL: {article_data.url if article_data.url else 'N/A'}")
    result = await ingestion_service.ingest_single_article(
        url=article_data.url, text_content=article_data.text_content, source_name=article_data.source_name
    ) # This service function now handles DISABLE_RAG internally too.
    if result["status"] == "error":
        logger.error(f"Ingestion failed for source '{article_data.source_name}': {result['message']}")
        raise HTTPException(status_code=400 if "content" in result.get('message','').lower() else 500, detail=result["message"])
    elif result["status"] == "warning":
        logger.warning(f"Ingestion for source '{article_data.source_name}' warning: {result['message']}")
        return JSONResponse(status_code=202, content=result)
    return schemas.IngestResponse(**result)


# --- Chat Endpoint (conditionally RAG-powered) ---
@app.post("/chat", tags=["Chat"])
async def stream_chat_response(
    chat_request: schemas.ChatRequest,
    x_session_id: Annotated[str | None, Header(convert_underscores=True)] = None
):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-Id header is required.")
    
    logger.info(f"Chat request for session_id: {x_session_id}, query: '{chat_request.query[:50]}...'")

    # Basic check for Redis (always needed)
    if not rag_service.redis_client:
        logger.error(f"Chat service unavailable for session {x_session_id}: Redis client missing.")
        async def error_stream_redis():
            yield f"data: {json.dumps({'error': 'Chat service session error.', 'final': True, 'type': 'error'})}\n\n"
        return StreamingResponse(error_stream_redis(), media_type="text/event-stream")

    # If RAG is enabled, check RAG-specific components
    if not settings.DISABLE_RAG:
        if not all([rag_service.query_embedding_model, rag_service.qdrant_client_rag, rag_service.generative_llm]):
            logger.error(f"Chat service: RAG ML components not ready for session {x_session_id}.")
            async def error_stream_ml():
                yield f"data: {json.dumps({'error': 'AI features temporarily unavailable.', 'final': True, 'type': 'error'})}\n\n"
            return StreamingResponse(error_stream_ml(), media_type="text/event-stream")

    # Call process_chat_query - it will handle DISABLE_RAG internally
    async def event_generator():
        # ... (your existing event_generator logic is fine here) ...
        try:
            async for content_chunk in rag_service.process_chat_query(chat_request.query, x_session_id):
                yield f"data: {json.dumps({'text': content_chunk, 'type': 'content'})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'type': 'event'})}\n\n"
        except Exception as e:
            logger.error(f"Error during chat streaming for session {x_session_id}: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': 'Unexpected error processing request.', 'final': True, 'type': 'error'})}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- Main Execution ---
if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server directly for development (RAG Disabled: {settings.DISABLE_RAG})...")
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST, port=settings.API_PORT,
        reload=settings.API_RELOAD, log_level=settings.LOG_LEVEL.lower()
    )