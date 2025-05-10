# app/services/rag_service.py
import json
from typing import List, Dict, AsyncGenerator, Optional
import traceback # Added here
import logging # Added logging

from sentence_transformers import SentenceTransformer # For query embedding
from qdrant_client import QdrantClient
import google.generativeai as genai
import redis.asyncio as redis # For chat history

from app.core.config import settings # Ensure settings is imported
from app.models.schemas import ChatMessage # Pydantic model for chat messages

# Initialize logger for this module
logger = logging.getLogger(__name__)

# --- Initialize Clients ---
query_embedding_model: Optional[SentenceTransformer] = None
qdrant_client_rag: Optional[QdrantClient] = None # Use a different name to avoid conflict if ingestion_service also defines qdrant_client
generative_llm: Optional[genai.GenerativeModel] = None
redis_client: Optional[redis.Redis] = None

# Attempt to import from ingestion_service (preferred)
try:
    from app.services.ingestion_service import embedding_model as ig_embedding_model
    from app.services.ingestion_service import qdrant_client as ig_qdrant_client

    if ig_embedding_model is None:
        logger.warning("rag_service: Embedding model from ingestion_service is None.")
        # Potentially trigger fallback or log critical error
    else:
        query_embedding_model = ig_embedding_model
        logger.info("rag_service: Successfully imported query_embedding_model from ingestion_service.")

    if ig_qdrant_client is None:
        logger.warning("rag_service: Qdrant client from ingestion_service is None.")
        # Potentially trigger fallback or log critical error
    else:
        qdrant_client_rag = ig_qdrant_client # Assign to our scoped variable
        logger.info("rag_service: Successfully imported qdrant_client from ingestion_service.")

    if query_embedding_model is None or qdrant_client_rag is None:
         raise ImportError("One or more components from ingestion_service were None.")

except ImportError as e:
    logger.warning(f"rag_service: Could not fully import from ingestion_service: {e}. Attempting fallback initialization for missing components.")
    # Fallback initialization for query_embedding_model if not loaded
    if query_embedding_model is None:
        try:
            logger.info(f"Fallback: Initializing query_embedding_model with {settings.SENTENCE_TRANSFORMER_MODEL_NAME}")
            query_embedding_model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL_NAME, trust_remote_code=True)
            logger.info("rag_service: Fallback - Re-initialized query_embedding_model.")
        except Exception as fallback_e_emb:
            logger.critical(f"rag_service: CRITICAL - Fallback initialization for query_embedding_model failed: {fallback_e_emb}", exc_info=True)
            query_embedding_model = None # Ensure it's None if failed

    # Fallback initialization for qdrant_client_rag if not loaded
    if qdrant_client_rag is None:
        try:
            logger.info(f"Fallback: Initializing qdrant_client_rag for Qdrant at {settings.QDRANT_URL}")
            qdrant_client_rag = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
            logger.info("rag_service: Fallback - Re-initialized qdrant_client_rag.")
        except Exception as fallback_e_qdrant:
            logger.critical(f"rag_service: CRITICAL - Fallback initialization for qdrant_client_rag failed: {fallback_e_qdrant}", exc_info=True)
            qdrant_client_rag = None # Ensure it's None if failed

# Initialize Gemini Pro model
try:
    logger.info(f"Initializing Gemini Pro model: {settings.GEMINI_MODEL_NAME}")
    genai.configure(api_key=settings.GOOGLE_API_KEY)
    generative_llm = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
    logger.info(f"rag_service: Gemini Pro model '{settings.GEMINI_MODEL_NAME}' initialized.")
except Exception as e:
    logger.critical(f"rag_service: CRITICAL - Error initializing Gemini Pro model: {e}", exc_info=True)
    generative_llm = None

# Initialize Redis client for chat history
try:
    logger.info(f"Initializing Redis client for chat history at {settings.REDIS_URL}")
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True) # decode_responses=True is important for strings
    # Test connection
    # await redis_client.ping() # This would require the function to be async or run in an event loop
    logger.info(f"rag_service: Redis client initialized for chat history.")
except Exception as e:
    logger.critical(f"rag_service: CRITICAL - Error initializing Redis client: {e}", exc_info=True)
    redis_client = None

# --- Core RAG Functions ---

async def get_query_embedding(query_text: str) -> Optional[List[float]]:
    """Generates embedding for the user query."""
    if not query_embedding_model:
        logger.error("get_query_embedding: Query embedding model not available.")
        return None
    try:
        # SentenceTransformer expects a list of texts
        embedding = query_embedding_model.encode([query_text], show_progress_bar=False)[0].tolist()
        logger.debug(f"get_query_embedding: Generated embedding for query (first 3 dims): {embedding[:3]}")
        return embedding
    except Exception as e:
        logger.error(f"get_query_embedding: Error generating query embedding: {e}", exc_info=True)
        return None

async def search_relevant_chunks(query_embedding: List[float], top_k: int = settings.TOP_K_RESULTS) -> List[Dict]:
    """Searches Qdrant for relevant document chunks."""
    if not qdrant_client_rag: # Use the rag specific client
        logger.error("search_relevant_chunks: Qdrant client (rag) not available.")
        return []
    if not query_embedding:
        logger.warning("search_relevant_chunks: No query embedding provided.")
        return []

    try:
        logger.debug(f"Searching Qdrant collection '{settings.VECTOR_DB_COLLECTION_NAME}' with top_k={top_k}")
        search_results = await qdrant_client_rag.search( # Qdrant client search is not async by default, check qdrant client version
            collection_name=settings.VECTOR_DB_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        ) # If using qdrant_client > 1.1.0, search is sync. For async, need to wrap or use specific async methods if qdrant provides.
          # For now, assuming sync search is acceptable or qdrant_client handles it.
          # If performance becomes an issue, investigate qdrant_client's async capabilities or run in thread pool.

        logger.debug(f"Found {len(search_results)} results from Qdrant.")

        formatted_results = []
        for hit in search_results:
            payload = hit.payload or {} # Ensure payload is not None
            formatted_results.append({
                "id": str(hit.id), # Ensure ID is string
                "score": float(hit.score), # Ensure score is float
                "text": payload.get("text", ""),
                "source_url": payload.get("source_url", "N/A"),
                "source_name": payload.get("source_name", "N/A"),
                "article_id": payload.get("article_id", "N/A")
            })
        return formatted_results
    except Exception as e:
        logger.error(f"Error searching Qdrant: {e}", exc_info=True)
        return []

def format_context_for_llm(retrieved_chunks: List[Dict]) -> str:
    """Formats the retrieved chunks into a string for the LLM prompt."""
    if not retrieved_chunks:
        return "No relevant context was found in the knowledge base for your query."

    context_str = "Here is some context from relevant news articles:\n\n"
    for i, chunk in enumerate(retrieved_chunks):
        context_str += f"--- Context Chunk {i+1} ---\n"
        if chunk.get("source_name") and chunk.get("source_name") != "N/A":
            context_str += f"Source: {chunk.get('source_name')}\n"
        # Only show actual URLs, not localfile:// placeholders
        if chunk.get("source_url") and chunk.get("source_url") != "N/A" and not chunk.get("source_url", "").startswith("localfile://"):
             context_str += f"URL: {chunk.get('source_url')}\n"
        context_str += f"Content: {chunk.get('text', '')}\n\n"

    logger.debug(f"format_context_for_llm: Formatted context string (first 100 chars): {context_str[:100]}")
    return context_str

def construct_llm_prompt(query: str, context: str, chat_history: List[ChatMessage]) -> str:
    """Constructs the full prompt for the Gemini LLM."""
    system_prompt = (
        "You are a helpful and concise AI assistant answering questions about news articles. "
        "Use ONLY the provided context to answer the user's question. "
        "If the information is not in the context or the context is not relevant, "
        "clearly state that you cannot answer based on the provided information or that the context doesn't cover the query. "
        "Do not make up information or answer from your general knowledge. Be factual. "
        "If asked about a previous part of the conversation, use the chat history provided below. "
        "Keep your answers to a few sentences unless specifically asked for more details."
    )

    history_str = ""
    if chat_history:
        history_str = "Chat History (most recent messages first):\n"
        # Show history in a more natural conversational order (recent last, but for prompt, LLM might prefer chronological)
        # For now, keeping the original order as it was:
        for msg in chat_history: # Iterate through the history as provided
            history_str += f"{msg.role.capitalize()}: {msg.content}\n"
        history_str += "\n"

    full_prompt = f"{system_prompt}\n\n"
    if history_str:
        full_prompt += f"{history_str}"
    full_prompt += f"Context from News Articles:\n{context}\n\n"
    full_prompt += f"User Question: {query}\n\nAssistant Answer:"

    logger.debug(f"construct_llm_prompt: Constructed prompt (first 200 chars): {full_prompt[:200]}")
    return full_prompt

async def get_llm_response_stream(prompt: str) -> AsyncGenerator[str, None]:
    """Gets a streaming response from the Gemini LLM."""
    if not generative_llm:
        logger.error("get_llm_response_stream: Generative LLM not available.")
        yield "Error: The AI model is currently unavailable. Please try again later."
        return

    logger.debug("get_llm_response_stream: Sending prompt to Gemini...")
    try:
        async for chunk in await generative_llm.generate_content_async(prompt, stream=True):
            if chunk.text:
                yield chunk.text
            # Optional: Log finish reason if available and useful
            # if chunk.candidates and chunk.candidates[0].finish_reason:
            #     logger.debug(f"LLM stream chunk finished. Reason: {chunk.candidates[0].finish_reason}")
    except Exception as e:
        logger.error(f"get_llm_response_stream: Error during Gemini API call: {e}", exc_info=True)
        yield "Error: Sorry, I encountered an issue while generating a response. Please check the logs."

# --- Chat History Management (Redis) ---
CHAT_HISTORY_KEY_PREFIX = "chat_history:"
# MAX_CHAT_HISTORY_LENGTH is now expected from settings.py (e.g., settings.MAX_CHAT_HISTORY_LENGTH)
# If not in settings, define it here: e.g., MAX_CHAT_HISTORY_LENGTH = 10

async def get_chat_history(session_id: str) -> List[ChatMessage]:
    """Retrieves chat history for a session from Redis."""
    if not redis_client:
        logger.error("get_chat_history: Redis client not available.")
        return []
    key = f"{CHAT_HISTORY_KEY_PREFIX}{session_id}"
    try:
        history_json_list = await redis_client.lrange(key, 0, -1)
        history = [ChatMessage.model_validate_json(item_json) for item_json in history_json_list]
        logger.debug(f"Retrieved {len(history)} messages for session {session_id}.")
        return history
    except Exception as e:
        logger.error(f"Error retrieving history for session {session_id} from key {key}: {e}", exc_info=True)
        return []

async def add_message_to_history(session_id: str, message: ChatMessage):
    """Adds a message to the chat history in Redis, trims old messages, and sets TTL."""
    if not redis_client:
        logger.error("add_message_to_history: Redis client not available.")
        return

    key = f"{CHAT_HISTORY_KEY_PREFIX}{session_id}"
    try:
        # Convert ChatMessage to JSON string for storage
        message_json = message.model_dump_json()

        # Use a pipeline for atomicity of RPUSH, LTRIM, and EXPIRE
        async with redis_client.pipeline() as pipe:
            pipe.rpush(key, message_json)
            # Ensure MAX_CHAT_HISTORY_LENGTH is defined. Let's use a default if not in settings.
            max_len = getattr(settings, 'MAX_CHAT_HISTORY_LENGTH', 10) * 2 # pairs of messages
            pipe.ltrim(key, -max_len, -1)
            pipe.expire(key, settings.REDIS_SESSION_TTL_SECONDS)
            await pipe.execute()

        logger.info(f"Added message to history for session {session_id}. Role: {message.role}. TTL set to {settings.REDIS_SESSION_TTL_SECONDS}s.")
    except Exception as e:
        logger.error(f"Error adding message to history for session {session_id} using key {key}: {e}", exc_info=True)

async def clear_chat_history(session_id: str):
    """Clears chat history for a session from Redis."""
    if not redis_client:
        logger.error("clear_chat_history: Redis client not available.")
        return
    key = f"{CHAT_HISTORY_KEY_PREFIX}{session_id}"
    try:
        await redis_client.delete(key)
        logger.info(f"Cleared history for session {session_id} (key: {key}).")
    except Exception as e:
        logger.error(f"Error clearing history for session {session_id} (key: {key}): {e}", exc_info=True)

# --- Main RAG Orchestration ---
async def process_chat_query(query: str, session_id: str) -> AsyncGenerator[str, None]:
    """Orchestrates the RAG pipeline for a given query and session."""
    logger.info(f"Processing chat query for session_id={session_id}. Query: '{query[:100]}...'")

    if not query or not query.strip():
        logger.warning(f"Empty query received for session_id={session_id}.")
        yield "It seems you sent an empty message. Please provide a question about the news articles."
        # Add a polite assistant message to history for empty query
        assistant_clarification = ChatMessage(role="assistant", content="I received an empty message. How can I help you regarding the news articles?")
        await add_message_to_history(session_id, assistant_clarification)
        return

    # 1. Add user's current query to history FIRST (so it's saved even if subsequent steps fail)
    user_message = ChatMessage(role="user", content=query)
    await add_message_to_history(session_id, user_message)

    # 2. Get chat history for prompt context (limited length)
    current_chat_history = await get_chat_history(session_id)
    # Limit history for prompt context to avoid overly long prompts
    # settings.HISTORY_FOR_PROMPT_COUNT or similar can be used
    prompt_history_count = getattr(settings, 'PROMPT_HISTORY_MESSAGES_COUNT', 6) # e.g., last 3 pairs
    prompt_history = current_chat_history[-prompt_history_count:]

    # 3. Generate query embedding
    query_emb = await get_query_embedding(query)
    if not query_emb:
        error_msg = "Error: I encountered an issue processing your query (embedding generation failed). Please try again."
        yield error_msg
        # Add this error to history as an assistant message
        error_msg_obj = ChatMessage(role="assistant", content=error_msg)
        await add_message_to_history(session_id, error_msg_obj)
        return

    # 4. Search for relevant chunks
    retrieved_chunks = await search_relevant_chunks(query_emb)
    if not retrieved_chunks:
        logger.info(f"No relevant chunks found for query in session {session_id}.")
        # No specific message yielded here yet, context_str will handle it.

    # 5. Format context
    context_str = format_context_for_llm(retrieved_chunks)

    # 6. Construct prompt
    llm_prompt = construct_llm_prompt(query, context_str, prompt_history)

    # 7. Get LLM response stream and add full response to history
    full_llm_response_parts = []
    try:
        async for response_chunk in get_llm_response_stream(llm_prompt):
            full_llm_response_parts.append(response_chunk)
            yield response_chunk
        full_llm_response = "".join(full_llm_response_parts)
        logger.info(f"Full LLM response collected for session {session_id} (length {len(full_llm_response)}).")
    except Exception as stream_err: # Should be caught by get_llm_response_stream, but as a safeguard
        logger.error(f"Error during LLM stream processing for session {session_id}: {stream_err}", exc_info=True)
        full_llm_response = "Error: I encountered an issue while generating the response stream."
        yield full_llm_response # Yield the error to the user

    # 8. Add final assistant response to history
    if full_llm_response and full_llm_response.strip():
        assistant_message = ChatMessage(role="assistant", content=full_llm_response.strip())
        await add_message_to_history(session_id, assistant_message)
    else:
        # This case might happen if LLM yields nothing or only whitespace, or if stream errored and yielded an error message.
        # If full_llm_response is an error message we yielded, it's already in history as user message.
        # We need to be careful not to add it twice or add an empty assistant message.
        # The current logic adds the yielded content to history, which is fine if it's an error message *from the LLM stream*.
        logger.warning(f"LLM produced an empty or whitespace-only response for session {session_id}. Not adding to history as new message unless it was an error message from stream.")
        if not full_llm_response.strip() and not "Error:" in full_llm_response: # If truly empty and not an error we sent
            no_answer_message = "I couldn't find a specific answer based on the provided context or your query. Could you try rephrasing or asking something else?"
            yield no_answer_message
            assistant_no_answer = ChatMessage(role="assistant", content=no_answer_message)
            await add_message_to_history(session_id, assistant_no_answer)


    logger.info(f"Finished processing chat query for session_id={session_id}.")