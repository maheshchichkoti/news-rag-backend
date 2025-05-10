# app/services/ingestion_service.py
import httpx
from bs4 import BeautifulSoup
from trafilatura import extract
# SentenceTransformer, QdrantClient, tiktoken will be conditionally imported/used
from qdrant_client import models as qdrant_models # Keep this for PointStruct if used in stubs
from qdrant_client.http.models import PointStruct

import uuid
import hashlib
from typing import List, Dict, Optional
import traceback
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Conditionally Initialize Clients ---
qdrant_client: Optional['QdrantClient'] = None
embedding_model: Optional['SentenceTransformer'] = None
EMBEDDING_DIM: int = 384
tokenizer: Optional['tiktoken.Encoding'] = None

if not settings.DISABLE_RAG:
    logger.info("RAG ENABLED: Initializing ingestion service components...")
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        import tiktoken

        logger.info("Initializing Qdrant client...")
        qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
            timeout=20,
        )
        logger.info(f"Qdrant client initialized. Target: {settings.QDRANT_URL}")

        logger.info(f"Loading embedding model: {settings.SENTENCE_TRANSFORMER_MODEL_NAME}...")
        embedding_model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL_NAME, trust_remote_code=True)
        retrieved_dim = embedding_model.get_sentence_embedding_dimension()
        if retrieved_dim is not None:
            EMBEDDING_DIM = retrieved_dim
        logger.info(f"Embedding model loaded. Dimension: {EMBEDDING_DIM}")

        logger.info("Initializing tokenizer: cl100k_base...")
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("Tokenizer cl100k_base initialized.")
        except:
            logger.warning("cl100k_base tokenizer failed, trying gpt2.")
            tokenizer = tiktoken.get_encoding("gpt2")
            logger.info("Tokenizer gpt2 initialized.")

    except ImportError as e:
        logger.critical(f"Failed to import core ML libraries for ingestion_service: {e}. RAG features will be impaired.", exc_info=True)
    except Exception as e:
        logger.critical(f"Error during RAG component initialization in ingestion_service: {e}", exc_info=True)
else:
    logger.warning("RAG DISABLED: Skipping initialization of ingestion service components.")

# --- Helper Functions ---
def get_text_tokenizer():
    if settings.DISABLE_RAG or tokenizer is None:
        logger.warning("Tokenizer accessed but RAG is disabled or tokenizer not initialized.")
        # Fallback or raise error. For assessment, maybe a dummy/no-op.
        class DummyTokenizer:
            def encode(self, text): return []
            def decode(self, tokens): return ""
        return DummyTokenizer()
    return tokenizer

def count_tokens(text: str) -> int:
    if settings.DISABLE_RAG or not text: return 0
    # ... (rest of your count_tokens, using get_text_tokenizer()) ...
    try:
        return len(get_text_tokenizer().encode(text))
    except Exception as e:
        logger.error(f"count_tokens: Error encoding text: {e}", exc_info=True); return 0


async def fetch_article_content(url: str) -> Optional[str]:
    # This function can remain largely the same as it doesn't depend on ML models
    # ... (your existing fetch_article_content logic) ...
    logger.info(f"Attempting to fetch URL: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive'
    }
    downloaded_html: Optional[str] = None
    response_obj: Optional[httpx.Response] = None

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers, verify=True) as client:
            logger.debug(f"Sending GET request to {url}...")
            try:
                response_obj = await client.get(url)
                logger.debug(f"Received response object for {url}: Status {response_obj.status_code}")
            except Exception as get_exc:
                logger.error(f"Exception during client.get({url}): {get_exc}", exc_info=True)
                return None

            if response_obj is not None:
                logger.debug(f"Response details for {url}: Status: {response_obj.status_code}, HTTP Version: {response_obj.http_version}, URL after redirects: {response_obj.url}")
                if response_obj.status_code == 200:
                    try:
                        raw_body = await response_obj.aread()
                        if isinstance(raw_body, bytes):
                            detected_encoding = response_obj.encoding or 'utf-8'
                            logger.debug(f"Byte response detected for {url}. Attempting decode with: {detected_encoding}")
                            try:
                                downloaded_html = raw_body.decode(detected_encoding, errors='replace')
                            except Exception:
                                logger.warning(f"Primary decode failed for {url}, trying utf-8 force.")
                                downloaded_html = raw_body.decode('utf-8', errors='replace')
                        else:
                            downloaded_html = raw_body
                        logger.info(f"Successfully downloaded/read HTML (Length: {len(downloaded_html) if downloaded_html else 'N/A'}) from: {url}")
                    except Exception as read_err:
                        logger.error(f"Error reading/decoding response content for {url}: {read_err}", exc_info=True)
                        return None
                else:
                    error_text_snippet = response_obj.text[:200] if response_obj.text else "No response text."
                    logger.warning(f"HTTP Error {response_obj.status_code} for {url}. Snippet: {error_text_snippet}")
                    return None # Important to return None on non-200
            else:
                logger.warning(f"client.get({url}) did not yield a response object.")
                return None
    except httpx.TimeoutException as e:
        logger.warning(f"httpx.TimeoutException for {url}: {e}"); return None
    except httpx.ConnectError as e:
        logger.warning(f"httpx.ConnectError for {url}: {e}"); return None
    except httpx.RequestError as e:
        logger.warning(f"httpx.RequestError (other) for {url}: {e.__class__.__name__} - {e}"); return None
    except Exception as e:
        logger.error(f"Generic error during HTTP fetch for {url}: {e.__class__.__name__} - {e}", exc_info=True); return None

    if downloaded_html and downloaded_html.strip():
        if settings.DISABLE_RAG: # Trafilatura can be heavy, skip if RAG disabled
            logger.warning("RAG is disabled, skipping trafilatura extraction for assessment mode.")
            # Return a small part of HTML or a placeholder to show download worked
            return downloaded_html[:500] + "... (content extraction skipped in RAG-disabled mode)"

        logger.debug(f"Proceeding to trafilatura for {url} (HTML length: {len(downloaded_html)})")
        content = extract(downloaded_html, include_comments=False, include_tables=False, output_format='txt', favor_precision=True, url=url)
        if content:
            content = "\n".join([line for line in content.splitlines() if line.strip()])
            logger.info(f"Trafilatura extracted (Length: {len(content)}) for: {url}"); return content
        else:
            logger.warning(f"Trafilatura failed for {url}. Trying BeautifulSoup.")
            try:
                soup = BeautifulSoup(downloaded_html, 'html.parser')
                # ... (rest of BS logic, or also conditionally skip)
                # For assessment mode, you might even simplify this further
                body_text_bs = soup.body.get_text(separator='\n',strip=True) if soup.body else ""
                body_text_bs = "\n".join([line for line in body_text_bs.splitlines() if line.strip()])
                logger.info(f"BS body fallback (Length: {len(body_text_bs)}) for: {url}"); return body_text_bs if body_text_bs else None
            except Exception as e_bs:
                logger.error(f"BS fallback error for {url}: {e_bs}", exc_info=True); return None
    else:
        logger.warning(f"Failed to download or HTML empty/whitespace for: {url}")
    return None


def chunk_text(text: str, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP) -> List[str]:
    if settings.DISABLE_RAG:
        logger.warning("Chunking attempt while RAG is disabled.")
        return [text[:chunk_size]] if text else [] # Return a dummy chunk
    # ... (rest of your chunk_text, using get_text_tokenizer()) ...
    logger.debug(f"chunk_text: Entered. Text length: {len(text)}, Chunk_size: {chunk_size}, Overlap: {chunk_overlap}")
    if not text: logger.warning("chunk_text: No text provided."); return []
    # tokenizer should be the dummy tokenizer if RAG is disabled
    current_tokenizer = get_text_tokenizer()
    try:
        tokens = current_tokenizer.encode(text)
    except Exception as e:
        logger.error(f"chunk_text: Failed to encode text: {e}", exc_info=True); return []

    if not tokens: logger.warning("chunk_text: No tokens from text."); return []
    # ... rest of chunking logic
    chunks = []
    current_pos = 0; max_tokens = len(tokens)
    while current_pos < max_tokens:
        end_pos = min(current_pos + chunk_size, max_tokens)
        chunk_text_content = current_tokenizer.decode(tokens[current_pos:end_pos])
        if chunk_text_content.strip(): chunks.append(chunk_text_content.strip())
        if end_pos == max_tokens: break
        step = chunk_size - chunk_overlap
        if step <= 0:
            logger.warning("chunk_text: Chunk size <= overlap. Advancing by 1 token to avoid infinite loop."); step = 1
        current_pos += step
    logger.info(f"chunk_text: Original (tokens: {len(tokens)}) chunked into {len(chunks)} chunks.")
    return chunks


def ensure_qdrant_collection() -> bool:
    if settings.DISABLE_RAG:
        logger.info("ensure_qdrant_collection: RAG disabled, skipping Qdrant check.")
        return True # Pretend it's fine
    if not qdrant_client: logger.error("ensure_qdrant_collection: Qdrant client not initialized."); return False
    if not embedding_model: logger.error("ensure_qdrant_collection: Embedding model not initialized."); return False
    # ... (rest of your ensure_qdrant_collection logic) ...
    try:
        logger.debug(f"Checking Qdrant collection '{settings.VECTOR_DB_COLLECTION_NAME}'.")
        qdrant_client.get_collection(collection_name=settings.VECTOR_DB_COLLECTION_NAME)
        logger.info(f"Collection '{settings.VECTOR_DB_COLLECTION_NAME}' exists. Expected Dim for new points: {EMBEDDING_DIM}")
        return True
    except Exception as e:
        is_not_found = ("not found" in str(e).lower() or \
                        "collectionnotfoundexception" in str(e).lower().replace("_","") or \
                        (hasattr(e, 'status_code') and e.status_code == 404))
        if is_not_found:
            logger.info(f"Collection '{settings.VECTOR_DB_COLLECTION_NAME}' not found. Creating...")
            try:
                qdrant_client.recreate_collection(
                    collection_name=settings.VECTOR_DB_COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(size=EMBEDDING_DIM, distance=qdrant_models.Distance.COSINE) # Use qdrant_models
                )
                logger.info(f"Collection '{settings.VECTOR_DB_COLLECTION_NAME}' created with dim {EMBEDDING_DIM}.")
                return True
            except Exception as ce:
                logger.exception(f"Error CREATING collection '{settings.VECTOR_DB_COLLECTION_NAME}': {ce}")
                return False
        else:
            logger.exception(f"Error CHECKING collection '{settings.VECTOR_DB_COLLECTION_NAME}': {e}")
            return False
    finally:
        logger.debug("ensure_qdrant_collection: Exiting.")


async def ingest_single_article(
    url: Optional[str]=None, text_content: Optional[str]=None,
    source_name: Optional[str]="Unknown", article_id_override: Optional[str]=None
) -> Dict:
    if settings.DISABLE_RAG:
        logger.warning(f"Ingest attempt for source '{source_name}' while RAG is disabled.")
        return {"status":"error", "message":"Ingestion and RAG features are disabled in this mode."}

    logger.info(f"ingest_single_article: Processing source='{source_name}', URL='{url if url else 'N/A (text provided)'}'")
    if not all([qdrant_client, embedding_model, tokenizer]):
        logger.error("ingest_single_article: Core RAG components not initialized.")
        return {"status":"error", "message":"Core RAG components not initialized."}
    # ... (rest of your ingest_single_article logic) ...
    if not ensure_qdrant_collection():
        logger.error("ingest_single_article: Qdrant collection setup failed.")
        return {"status":"error", "message":"Qdrant collection error."}

    actual_content: Optional[str] = None
    if url and not text_content:
        actual_content = await fetch_article_content(url)
    elif text_content:
        actual_content = text_content

    if not (actual_content and actual_content.strip()):
        message = f"Content is empty or whitespace for URL: {url}" if url else "Provided text_content is empty or whitespace."
        logger.warning(f"ingest_single_article: {message}")
        return {"status":"error", "message": message}

    article_chunks = chunk_text(actual_content)
    if not article_chunks:
        logger.warning(f"ingest_single_article: No chunks generated for source='{source_name}', URL='{url}'")
        return {"status":"error", "message":"No chunks could be generated from the content."}

    base_article_id_for_payload = article_id_override or \
                                  (hashlib.sha256(url.encode('utf-8')).hexdigest() if url else str(uuid.uuid4()))
    points = []
    for i, chunk in enumerate(filter(str.strip, article_chunks)):
        try:
            emb = embedding_model.encode([chunk], show_progress_bar=False)[0].tolist()
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk, "source_url": url or "N/A", "source_name": source_name,
                "article_id": base_article_id_for_payload, "chunk_index": i,
                "original_text_token_count": count_tokens(actual_content),
                "chunk_token_count": count_tokens(chunk)
            }
            points.append(PointStruct(id=point_id, vector=emb, payload=payload))
        except Exception as e:
            logger.error(f"Error processing chunk {i} for article_id {base_article_id_for_payload}: {e}", exc_info=True)

    if not points:
        logger.warning(f"No valid points for article_id {base_article_id_for_payload}.")
        return {"status":"warning", "message":f"No valid points for article_id {base_article_id_for_payload}."}
    try:
        qdrant_client.upsert(collection_name=settings.VECTOR_DB_COLLECTION_NAME, points=points, wait=True)
        msg = f"Upserted {len(points)} chunks for article_id {base_article_id_for_payload}"
        logger.info(f"ingest_single_article: {msg}")
        return {"status":"success","message":msg,"article_id":base_article_id_for_payload,"num_chunks":len(points)}
    except Exception as e:
        logger.exception(f"Qdrant upsert error for article_id {base_article_id_for_payload}: {e}")
        return {"status":"error","message":f"Qdrant upsert failed: {e}"}


# --- Module Import Finalization ---
if __name__ != "__main__":
    if settings.DISABLE_RAG:
        logger.warning("ingestion_service.py: Module imported, RAG features DISABLED.")
    else:
        logger.info("ingestion_service.py: Module imported. Performing RAG component checks...")
        # ... (your existing initialization_ok checks)
        initialization_ok = True
        if not qdrant_client: logger.critical("Qdrant client failed to initialize."); initialization_ok = False
        if not embedding_model: logger.critical("Embedding model failed to initialize."); initialization_ok = False
        if not tokenizer: logger.critical("Tokenizer failed to initialize."); initialization_ok = False

        if initialization_ok:
            logger.info("Core RAG components (Qdrant, Embedder, Tokenizer) seem initialized.")
            if not ensure_qdrant_collection():
                logger.critical("CRITICAL: Failed to ensure Qdrant collection post-import.")
            else:
                logger.info("Qdrant collection confirmed ready post-import.")
        else:
            logger.critical("One or more core RAG components failed to initialize.")
else:
    logger.info("ingestion_service.py: Script run directly.")