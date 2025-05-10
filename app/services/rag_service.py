# app/services/rag_service.py
import json
from typing import List, Dict, AsyncGenerator, Optional
import traceback
import logging
import redis.asyncio as redis # Always needed

from app.core.config import settings
from app.models.schemas import ChatMessage

logger = logging.getLogger(__name__)

# --- Conditionally Initialize Clients ---
query_embedding_model: Optional['SentenceTransformer'] = None
qdrant_client_rag: Optional['QdrantClient'] = None
generative_llm: Optional['genai.GenerativeModel'] = None
# redis_client is initialized unconditionally below

# Initialize Redis client (always needed for session management)
redis_client: Optional[redis.Redis] = None
try:
    logger.info(f"RAG_SERVICE: Initializing Redis client at {settings.REDIS_URL}")
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info("RAG_SERVICE: Redis client initialized successfully.")
except Exception as e:
    logger.critical("RAG_SERVICE: CRITICAL - Error initializing Redis client", exc_info=True)

if not settings.DISABLE_RAG:
    logger.info("RAG ENABLED: Initializing RAG_SERVICE ML/AI components...")
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        import google.generativeai as genai

        # Attempt to get models from ingestion_service if they were initialized there
        try:
            from app.services.ingestion_service import embedding_model as ig_embedding_model, \
                                                    qdrant_client as ig_qdrant_client
            if ig_embedding_model:
                query_embedding_model = ig_embedding_model
                logger.info("RAG_SERVICE: Using embedding_model from ingestion_service.")
            if ig_qdrant_client:
                qdrant_client_rag = ig_qdrant_client
                logger.info("RAG_SERVICE: Using qdrant_client from ingestion_service.")
        except ImportError:
            logger.warning("RAG_SERVICE: Could not import components from ingestion_service. Will attempt direct init.")

        if not query_embedding_model: # If not obtained from ingestion_service
            logger.info(f"RAG_SERVICE: Initializing query_embedding_model with {settings.SENTENCE_TRANSFORMER_MODEL_NAME}")
            query_embedding_model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL_NAME, trust_remote_code=True)
        if not qdrant_client_rag: # If not obtained from ingestion_service
            logger.info(f"RAG_SERVICE: Initializing qdrant_client_rag for Qdrant at {settings.QDRANT_URL}")
            qdrant_client_rag = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)

        logger.info(f"RAG_SERVICE: Initializing Gemini Pro model: {settings.GEMINI_MODEL_NAME}")
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        generative_llm = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
        logger.info("RAG_SERVICE: All RAG ML/AI components initialized.")

    except ImportError as e_import:
        logger.critical(f"RAG_SERVICE: Could not import RAG dependencies: {e_import}. RAG features will be non-functional.", exc_info=True)
    except Exception as e_init:
        logger.critical(f"RAG_SERVICE: Error initializing RAG components: {e_init}", exc_info=True)
else:
    logger.warning("RAG DISABLED: Skipping initialization of RAG_SERVICE ML/AI components.")

# --- Core RAG Functions (will be stubs if RAG disabled) ---
async def get_query_embedding(query_text: str) -> Optional[List[float]]:
    if settings.DISABLE_RAG or not query_embedding_model:
        logger.debug("get_query_embedding: RAG disabled or model unavailable.")
        return None
    # ... (your actual get_query_embedding logic)
    try:
        embedding = query_embedding_model.encode([query_text], show_progress_bar=False)[0].tolist()
        return embedding
    except Exception as e: logger.error(f"Error generating query embedding: {e}", exc_info=True); return None


async def search_relevant_chunks(query_embedding: List[float], top_k: int = settings.TOP_K_RESULTS) -> List[Dict]:
    if settings.DISABLE_RAG or not qdrant_client_rag:
        logger.debug("search_relevant_chunks: RAG disabled or Qdrant client unavailable.")
        return []
    # ... (your actual search_relevant_chunks logic)
    if not query_embedding: return []
    try:
        search_results = qdrant_client_rag.search( # Assuming sync search for now
            collection_name=settings.VECTOR_DB_COLLECTION_NAME, query_vector=query_embedding,
            limit=top_k, with_payload=True
        )
        # ... (format results)
        return [{"id": str(hit.id), "score": float(hit.score), **(hit.payload or {})} for hit in search_results]
    except Exception as e: logger.error(f"Error searching Qdrant: {e}", exc_info=True); return []


def format_context_for_llm(retrieved_chunks: List[Dict]) -> str:
    if settings.DISABLE_RAG: return "Context formatting skipped (RAG disabled)."
    # ... (your actual format_context_for_llm logic)
    if not retrieved_chunks: return "No relevant context found."
    # ... (build context_str)
    context_str = "Context:\n" + "\n".join([f"Chunk {i+1}: {chunk.get('text', '')}" for i, chunk in enumerate(retrieved_chunks)])
    return context_str

def construct_llm_prompt(query: str, context: str, chat_history: List[ChatMessage]) -> str:
    if settings.DISABLE_RAG: return f"LLM Prompt construction skipped. User query: {query}"
    # ... (your actual construct_llm_prompt logic)
    # Simplified for brevity
    return f"System: Answer based on context.\nHistory: {chat_history}\nContext: {context}\nQuery: {query}\nAnswer:"

async def get_llm_response_stream(prompt: str) -> AsyncGenerator[str, None]:
    if settings.DISABLE_RAG or not generative_llm:
        logger.debug("get_llm_response_stream: RAG disabled or LLM unavailable.")
        yield "AI model interaction is disabled in this mode."
        return
    # ... (your actual get_llm_response_stream logic)
    try:
        async for chunk in await generative_llm.generate_content_async(prompt, stream=True):
            if chunk.text: yield chunk.text
    except Exception as e:
        logger.error(f"Error during Gemini API call: {e}", exc_info=True)
        yield "Error: Issue generating AI response."


# --- Chat History Management (Redis - always active) ---
# ... (Your get_chat_history, add_message_to_history, clear_chat_history functions are fine)
# Ensure they use the `redis_client` initialized at the top of this file.
CHAT_HISTORY_KEY_PREFIX = "chat_history:"
async def get_chat_history(session_id: str) -> List[ChatMessage]: # Copied from your version
    if not redis_client: logger.error("get_chat_history: Redis client not available."); return []
    key = f"{CHAT_HISTORY_KEY_PREFIX}{session_id}"
    try:
        history_json_list = await redis_client.lrange(key, 0, -1)
        history = [ChatMessage.model_validate_json(item_json) for item_json in history_json_list]
        logger.debug(f"Retrieved {len(history)} messages for session {session_id}.")
        return history
    except Exception as e: logger.error(f"Error retrieving history for session {session_id} (key {key}): {e}", exc_info=True); return []

async def add_message_to_history(session_id: str, message: ChatMessage): # Copied from your version
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
    except Exception as e: logger.error(f"Error adding message to history for session {session_id} (key {key}): {e}", exc_info=True)

async def clear_chat_history(session_id: str): # Copied from your version
    if not redis_client: logger.error("clear_chat_history: Redis client not available."); return
    key = f"{CHAT_HISTORY_KEY_PREFIX}{session_id}"
    try:
        await redis_client.delete(key)
        logger.info(f"Cleared history for session {session_id} (key: {key}).")
    except Exception as e: logger.error(f"Error clearing history for session {session_id} (key: {key}): {e}", exc_info=True)


# --- Main RAG Orchestration ---
async def process_chat_query(query: str, session_id: str) -> AsyncGenerator[str, None]:
    logger.info(f"RAG_SERVICE: Processing chat query for session_id={session_id}. Query: '{query[:50]}...'")

    if not query or not query.strip(): # Handle empty query first
        logger.warning(f"RAG_SERVICE: Empty query for session_id={session_id}.")
        msg = "It seems you sent an empty message. Please type a question."
        yield msg
        if redis_client: await add_message_to_history(session_id, ChatMessage(role="assistant", content=msg))
        return

    # Add user message to history (always, even if RAG is disabled)
    if redis_client: await add_message_to_history(session_id, ChatMessage(role="user", content=query))

    if settings.DISABLE_RAG:
        logger.warning(f"RAG_SERVICE: Chat query for session {session_id} but RAG is disabled.")
        disabled_msg = "I am currently in a restricted mode for this demo and cannot provide AI-powered answers. Basic session and history are active."
        yield disabled_msg
        if redis_client: await add_message_to_history(session_id, ChatMessage(role="assistant", content=disabled_msg))
        return

    # --- Full RAG pipeline (only if not disabled and components are ready) ---
    if not all([query_embedding_model, qdrant_client_rag, generative_llm]):
        logger.error(f"RAG_SERVICE: Cannot process chat for session {session_id} - one or more RAG ML components are not available.")
        err_msg = "Apologies, the AI components required for a full answer are not currently available. Please try again later."
        yield err_msg
        if redis_client: await add_message_to_history(session_id, ChatMessage(role="assistant", content=err_msg))
        return

    current_chat_history = await get_chat_history(session_id) # Already includes current user query
    prompt_history_count = getattr(settings, 'PROMPT_HISTORY_MESSAGES_COUNT', 6)
    # Exclude the last user message we just added if it's for the prompt context of *that same message*
    prompt_history = current_chat_history[-(prompt_history_count + 1) : -1] if len(current_chat_history) > 1 else []


    query_emb = await get_query_embedding(query)
    if not query_emb:
        error_msg = "Error: Could not process your query (embedding failed)."
        yield error_msg
        if redis_client: await add_message_to_history(session_id, ChatMessage(role="assistant", content=error_msg))
        return

    retrieved_chunks = await search_relevant_chunks(query_emb)
    context_str = format_context_for_llm(retrieved_chunks)
    llm_prompt = construct_llm_prompt(query, context_str, prompt_history)

    full_llm_response = ""
    try:
        async for response_chunk in get_llm_response_stream(llm_prompt):
            full_llm_response += response_chunk
            yield response_chunk
    except Exception as e: # Should be caught in get_llm_response_stream, but as safety
        logger.error(f"RAG_SERVICE: Stream error in process_chat_query: {e}", exc_info=True)
        error_msg = "An unexpected error occurred during the AI response generation."
        full_llm_response = error_msg # So it gets added to history
        yield error_msg


    if full_llm_response and full_llm_response.strip():
        if redis_client: await add_message_to_history(session_id, ChatMessage(role="assistant", content=full_llm_response.strip()))
    elif not "Error:" in full_llm_response: # If truly empty and not an error message we generated
        logger.warning(f"RAG_SERVICE: LLM produced an empty or whitespace-only response for session {session_id}.")
        no_answer_msg = "I couldn't find a specific answer based on the provided context. Please try rephrasing."
        yield no_answer_msg
        if redis_client: await add_message_to_history(session_id, ChatMessage(role="assistant", content=no_answer_msg))

    logger.info(f"RAG_SERVICE: Finished processing chat query for session_id={session_id}.")