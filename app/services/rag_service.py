# app/services/rag_service.py
import json
from typing import List, Dict, AsyncGenerator, Optional
import traceback
import logging

# ML/AI libraries will be conditionally imported
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# import google.generativeai as genai
import redis.asyncio as redis

from app.core.config import settings
from app.models.schemas import ChatMessage

logger = logging.getLogger(__name__)

# --- Initialize Clients ---
query_embedding_model: Optional['SentenceTransformer'] = None
qdrant_client_rag: Optional['QdrantClient'] = None
generative_llm: Optional['genai.GenerativeModel'] = None
redis_client: Optional[redis.Redis] = None

# Initialize Redis client (always needed for session management)
try:
    logger.info(f"Initializing Redis client for chat history at {settings.REDIS_URL}")
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info("Redis client for chat history initialized.")
except Exception as e:
    logger.critical("CRITICAL - Error initializing Redis client", exc_info=True)
    # If Redis fails, chat history and sessions will not work. Consider how to handle.

if not settings.DISABLE_RAG:
    logger.info("RAG ENABLED: Initializing RAG service components...")
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        import google.generativeai as genai

        # Attempt to import from ingestion_service (preferred for consistency)
        try:
            from app.services.ingestion_service import embedding_model as ig_embedding_model
            from app.services.ingestion_service import qdrant_client as ig_qdrant_client

            if ig_embedding_model:
                query_embedding_model = ig_embedding_model
                logger.info("RAG service: Using query_embedding_model from ingestion_service.")
            else:
                logger.warning("RAG service: Embedding model from ingestion_service is None. Fallback might be attempted if coded.")

            if ig_qdrant_client:
                qdrant_client_rag = ig_qdrant_client
                logger.info("RAG service: Using qdrant_client from ingestion_service.")
            else:
                logger.warning("RAG service: Qdrant client from ingestion_service is None. Fallback might be attempted if coded.")

            if query_embedding_model is None or qdrant_client_rag is None:
                logger.warning("One or more components from ingestion_service were None. Attempting RAG service fallback initializations if needed.")

        except ImportError as e_import:
            logger.warning(f"Could not import from ingestion_service for RAG components: {e_import}. Attempting direct initialization.")

        # Fallback/Direct initialization for query_embedding_model if not loaded
        if query_embedding_model is None:
            logger.info(f"RAG service: Initializing query_embedding_model with {settings.SENTENCE_TRANSFORMER_MODEL_NAME}")
            query_embedding_model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL_NAME, trust_remote_code=True)
            logger.info("RAG service: Initialized query_embedding_model directly.")

        # Fallback/Direct initialization for qdrant_client_rag if not loaded
        if qdrant_client_rag is None:
            logger.info(f"RAG service: Initializing qdrant_client_rag for Qdrant at {settings.QDRANT_URL}")
            qdrant_client_rag = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
            logger.info("RAG service: Initialized qdrant_client_rag directly.")

        # Initialize Gemini Pro model
        logger.info(f"Initializing Gemini Pro model: {settings.GEMINI_MODEL_NAME}")
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        generative_llm = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
        logger.info(f"RAG service: Gemini Pro model '{settings.GEMINI_MODEL_NAME}' initialized.")

    except ImportError as e:
        logger.critical(f"Failed to import core ML/AI libraries for rag_service: {e}. RAG features will be impaired.", exc_info=True)
    except Exception as e:
        logger.critical(f"Error during RAG component initialization in rag_service: {e}", exc_info=True)
else:
    logger.warning("RAG DISABLED: Skipping initialization of RAG service ML components.")


# --- Core RAG Functions (conditionally functional) ---
async def get_query_embedding(query_text: str) -> Optional[List[float]]:
    if settings.DISABLE_RAG or not query_embedding_model:
        logger.warning("get_query_embedding called but RAG disabled or model unavailable.")
        return None
    # ... (rest of your get_query_embedding) ...
    try:
        embedding = query_embedding_model.encode([query_text], show_progress_bar=False)[0].tolist()
        logger.debug(f"Generated embedding for query (first 3 dims): {embedding[:3]}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}", exc_info=True)
        return None


async def search_relevant_chunks(query_embedding: List[float], top_k: int = settings.TOP_K_RESULTS) -> List[Dict]:
    if settings.DISABLE_RAG or not qdrant_client_rag:
        logger.warning("search_relevant_chunks called but RAG disabled or Qdrant client unavailable.")
        return []
    if not query_embedding:
        logger.warning("search_relevant_chunks: No query embedding provided.")
        return []
    # ... (rest of your search_relevant_chunks) ...
    try:
        logger.debug(f"Searching Qdrant collection '{settings.VECTOR_DB_COLLECTION_NAME}' with top_k={top_k}")
        # Assuming sync search for simplicity, adapt if qdrant_client has async methods you prefer
        search_results = qdrant_client_rag.search(
            collection_name=settings.VECTOR_DB_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        logger.debug(f"Found {len(search_results)} results from Qdrant.")
        formatted_results = []
        for hit in search_results:
            payload = hit.payload or {}
            formatted_results.append({
                "id": str(hit.id), "score": float(hit.score), "text": payload.get("text", ""),
                "source_url": payload.get("source_url", "N/A"),
                "source_name": payload.get("source_name", "N/A"),
                "article_id": payload.get("article_id", "N/A")
            })
        return formatted_results
    except Exception as e:
        logger.error(f"Error searching Qdrant: {e}", exc_info=True)
        return []


def format_context_for_llm(retrieved_chunks: List[Dict]) -> str:
    if settings.DISABLE_RAG: return "Context formatting skipped (RAG disabled)."
    # ... (rest of your format_context_for_llm) ...
    if not retrieved_chunks:
        return "No relevant context was found in the knowledge base for your query."
    context_str = "Here is some context from relevant news articles:\n\n"
    # ... (your loop)
    for i, chunk in enumerate(retrieved_chunks):
        context_str += f"--- Context Chunk {i+1} ---\n"
        if chunk.get("source_name") and chunk.get("source_name") != "N/A":
            context_str += f"Source: {chunk.get('source_name')}\n"
        if chunk.get("source_url") and chunk.get("source_url") != "N/A" and not chunk.get("source_url", "").startswith("localfile://"):
             context_str += f"URL: {chunk.get('source_url')}\n"
        context_str += f"Content: {chunk.get('text', '')}\n\n"
    logger.debug(f"Formatted context string (first 100 chars): {context_str[:100]}")
    return context_str


def construct_llm_prompt(query: str, context: str, chat_history: List[ChatMessage]) -> str:
    if settings.DISABLE_RAG: return f"LLM Prompt construction skipped (RAG disabled). User query: {query}"
    # ... (rest of your construct_llm_prompt) ...
    system_prompt = (
        "You are a helpful AI assistant for news. Use ONLY provided context. "
        "If no context or irrelevant, state you cannot answer from it. Be factual."
    ) # Shortened for brevity
    history_str = ""
    if chat_history:
        history_str = "Chat History:\n"
        for msg in chat_history: history_str += f"{msg.role.capitalize()}: {msg.content}\n"
        history_str += "\n"
    full_prompt = f"{system_prompt}\n\n{history_str}Context from News Articles:\n{context}\n\nUser Question: {query}\n\nAssistant Answer:"
    logger.debug(f"Constructed prompt (first 200 chars): {full_prompt[:200]}")
    return full_prompt

async def get_llm_response_stream(prompt: str) -> AsyncGenerator[str, None]:
    if settings.DISABLE_RAG or not generative_llm:
        logger.warning("get_llm_response_stream called but RAG disabled or LLM unavailable.")
        yield "AI model interaction is disabled in this mode."
        return
    # ... (rest of your get_llm_response_stream) ...
    logger.debug("Sending prompt to Gemini...")
    try:
        async for chunk in await generative_llm.generate_content_async(prompt, stream=True):
            if chunk.text: yield chunk.text
    except Exception as e:
        logger.error(f"Error during Gemini API call: {e}", exc_info=True)
        yield "Error: Issue generating AI response."


# --- Chat History Management (Redis - always active) ---
CHAT_HISTORY_KEY_PREFIX = "chat_history:"
# ... (get_chat_history, add_message_to_history, clear_chat_history remain the same as they only use Redis) ...
async def get_chat_history(session_id: str) -> List[ChatMessage]:
    if not redis_client: logger.error("get_chat_history: Redis client not available."); return []
    key = f"{CHAT_HISTORY_KEY_PREFIX}{session_id}"
    try:
        history_json_list = await redis_client.lrange(key, 0, -1)
        history = [ChatMessage.model_validate_json(item_json) for item_json in history_json_list]
        logger.debug(f"Retrieved {len(history)} messages for session {session_id}.")
        return history
    except Exception as e:
        logger.error(f"Error retrieving history for session {session_id} (key {key}): {e}", exc_info=True); return []

async def add_message_to_history(session_id: str, message: ChatMessage):
    if not redis_client: logger.error("add_message_to_history: Redis client not available."); return
    key = f"{CHAT_HISTORY_KEY_PREFIX}{session_id}"
    try:
        message_json = message.model_dump_json()
        async with redis_client.pipeline() as pipe:
            pipe.rpush(key, message_json)
            max_len = getattr(settings, 'MAX_CHAT_HISTORY_LENGTH', 10) * 2
            pipe.ltrim(key, -max_len, -1)
            pipe.expire(key, settings.REDIS_SESSION_TTL_SECONDS)
            await pipe.execute()
        logger.info(f"Added message to history for session {session_id}. Role: {message.role}. TTL set.")
    except Exception as e:
        logger.error(f"Error adding message to history for session {session_id} (key {key}): {e}", exc_info=True)

async def clear_chat_history(session_id: str):
    if not redis_client: logger.error("clear_chat_history: Redis client not available."); return
    key = f"{CHAT_HISTORY_KEY_PREFIX}{session_id}"
    try:
        await redis_client.delete(key)
        logger.info(f"Cleared history for session {session_id} (key: {key}).")
    except Exception as e:
        logger.error(f"Error clearing history for session {session_id} (key: {key}): {e}", exc_info=True)


# --- Main RAG Orchestration ---
async def process_chat_query(query: str, session_id: str) -> AsyncGenerator[str, None]:
    logger.info(f"Processing chat query for session_id={session_id}. Query: '{query[:100]}...'")

    if not query or not query.strip():
        logger.warning(f"Empty query for session_id={session_id}.")
        msg = "It seems you sent an empty message. Please type a question."
        yield msg
        await add_message_to_history(session_id, ChatMessage(role="assistant", content=msg))
        return

    # Add user message to history immediately
    await add_message_to_history(session_id, ChatMessage(role="user", content=query))

    if settings.DISABLE_RAG:
        logger.warning(f"Chat query for session {session_id} but RAG is disabled.")
        disabled_msg = "I am currently in a restricted mode for this demo and cannot provide AI-powered answers. Basic session and history are active."
        yield disabled_msg
        await add_message_to_history(session_id, ChatMessage(role="assistant", content=disabled_msg))
        return

    # --- Full RAG pipeline (only if not disabled) ---
    current_chat_history = await get_chat_history(session_id)
    prompt_history_count = getattr(settings, 'PROMPT_HISTORY_MESSAGES_COUNT', 6)
    prompt_history = current_chat_history[-prompt_history_count:]

    query_emb = await get_query_embedding(query)
    if not query_emb:
        error_msg = "Error: Could not process your query (embedding failed)."
        yield error_msg
        await add_message_to_history(session_id, ChatMessage(role="assistant", content=error_msg))
        return

    retrieved_chunks = await search_relevant_chunks(query_emb)
    context_str = format_context_for_llm(retrieved_chunks)
    llm_prompt = construct_llm_prompt(query, context_str, prompt_history)

    full_llm_response_parts = []
    final_response_type = "success" # Assume success initially
    try:
        async for response_chunk in get_llm_response_stream(llm_prompt):
            if "Error:" in response_chunk: # Check if the stream itself is sending an error
                final_response_type = "error_from_llm"
            full_llm_response_parts.append(response_chunk)
            yield response_chunk
        full_llm_response = "".join(full_llm_response_parts)
    except Exception as stream_err:
        logger.error(f"Unhandled error during LLM stream processing for session {session_id}: {stream_err}", exc_info=True)
        full_llm_response = "Critical Error: The AI response stream encountered an unexpected issue."
        final_response_type = "critical_stream_error"
        yield full_llm_response # Yield the critical error to the user

    logger.info(f"LLM response collected for session {session_id} (length {len(full_llm_response)}). Type: {final_response_type}")

    if full_llm_response and full_llm_response.strip():
        await add_message_to_history(session_id, ChatMessage(role="assistant", content=full_llm_response.strip()))
    elif final_response_type == "success": # LLM returned empty but no explicit error
        logger.warning(f"LLM produced empty response for session {session_id} (no explicit error from stream).")
        no_answer_msg = "I couldn't find a specific answer based on the provided context. Please try rephrasing."
        yield no_answer_msg
        await add_message_to_history(session_id, ChatMessage(role="assistant", content=no_answer_msg))
    # If it was an error message already yielded and part of full_llm_response, it's already in history.

    logger.info(f"Finished processing chat query for session_id={session_id}.")