# app/services/ingestion_service.py
import httpx
from bs4 import BeautifulSoup
# trafilatura, SentenceTransformer, QdrantClient, tiktoken conditionally imported
from qdrant_client import models as qdrant_models # For PointStruct if used in stubs
from qdrant_client.http.models import PointStruct

import uuid
import hashlib
from typing import List, Dict, Optional
import traceback
import logging

from app.core.config import settings # This now has the correct DISABLE_RAG

logger = logging.getLogger(__name__)

# --- Conditionally Initialize Clients ---
# These will remain None if settings.DISABLE_RAG is True
qdrant_client: Optional['QdrantClient'] = None
embedding_model: Optional['SentenceTransformer'] = None
EMBEDDING_DIM: int = 384 # Default, actual value set if model loads
tokenizer: Optional['tiktoken.Encoding'] = None
trafilatura_extract_func = None # Placeholder for trafilatura.extract

if not settings.DISABLE_RAG:
    logger.info("RAG ENABLED: Initializing ingestion_service components...")
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        import tiktoken
        from trafilatura import extract as trafilatura_extract_actual
        trafilatura_extract_func = trafilatura_extract_actual # Assign to our placeholder

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
        except:
            logger.warning("cl100k_base tokenizer failed, trying gpt2.")
            tokenizer = tiktoken.get_encoding("gpt2")
        logger.info("Tokenizer initialized.")
        logger.info("Ingestion_service RAG components initialized.")

    except ImportError as e_import:
        logger.critical(f"INGESTION_SERVICE: Could not import RAG dependencies: {e_import}. RAG features will be non-functional.", exc_info=True)
    except Exception as e_init:
        logger.critical(f"INGESTION_SERVICE: Error initializing RAG components: {e_init}", exc_info=True)
else:
    logger.warning("RAG DISABLED: Skipping initialization of ingestion_service ML/AI components.")

# --- Helper Functions ---
def get_text_tokenizer():
    if settings.DISABLE_RAG or tokenizer is None:
        logger.debug("Tokenizer accessed but RAG disabled or tokenizer not init.")
        class DummyTokenizer: # Simple stub
            def encode(self, text): return list(text) # Simplistic, replace if causes issues
            def decode(self, tokens): return "".join(tokens)
        return DummyTokenizer()
    return tokenizer

def count_tokens(text: str) -> int:
    if not text: return 0
    # When RAG is disabled, get_text_tokenizer() returns a dummy.
    # The dummy's encode might not be accurate for token counting,
    # so this count will be for the dummy if RAG is off.
    try:
        return len(get_text_tokenizer().encode(text))
    except Exception as e:
        logger.error(f"count_tokens: Error encoding text: {e}", exc_info=True); return len(text) # Fallback to char length

async def fetch_article_content(url: str) -> Optional[str]:
    # ... (Your existing fetch_article_content logic is mostly fine)
    # ... just ensure that if it calls trafilatura, it uses trafilatura_extract_func
    logger.info(f"Attempting to fetch URL: {url}")
    # ... (headers and try-except block for httpx.AsyncClient)
    # Inside the success block after getting downloaded_html:
    if downloaded_html and downloaded_html.strip():
        if settings.DISABLE_RAG or not trafilatura_extract_func:
            logger.warning("RAG is disabled or trafilatura not loaded, skipping full extraction.")
            return downloaded_html[:1000] + "... (full content extraction skipped)" # Return snippet

        content = trafilatura_extract_func(downloaded_html, include_comments=False, include_tables=False, output_format='txt', favor_precision=True, url=url)
        # ... (rest of your trafilatura/BeautifulSoup fallback logic) ...
    # ... (your existing fetch_article_content logic) ...
    # Make sure all paths return. The one you provided was good.
    # Simplified version for this paste:
    # ... (your existing fetch_article_content code, ensuring it uses `trafilatura_extract_func` if RAG is enabled)
    # For the purpose of this update, I'll assume your existing logic is fine but point out trafilatura usage.
    # Example part:
    # if downloaded_html and downloaded_html.strip():
    #     if not settings.DISABLE_RAG and trafilatura_extract_func:
    #         content = trafilatura_extract_func(downloaded_html, ...)
    #     else: # RAG disabled or trafilatura not available
    #         content = downloaded_html[:500] # Or some other stub
    # ...
    # This function is complex, ensure all paths are covered by the RAG disable logic if necessary.
    # For now, I'll keep your provided structure.
    # The key is that trafilatura_extract_func will be None if RAG is disabled.
    # So you'd check `if trafilatura_extract_func:`
    #
    # --- Start of your fetch_article_content ----
    # This is copied from your provided code, with minor adjustment for trafilatura_extract_func
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
            response_obj = await client.get(url)
            if response_obj.status_code == 200:
                raw_body = await response_obj.aread()
                if isinstance(raw_body, bytes):
                    detected_encoding = response_obj.encoding or 'utf-8'
                    try: downloaded_html = raw_body.decode(detected_encoding, errors='replace')
                    except: downloaded_html = raw_body.decode('utf-8', errors='replace')
                else: downloaded_html = raw_body
            else:
                logger.warning(f"HTTP Error {response_obj.status_code} for {url}.")
                return None
    except Exception as e:
        logger.error(f"Generic error during HTTP fetch for {url}: {e}", exc_info=True); return None

    if downloaded_html and downloaded_html.strip():
        if not settings.DISABLE_RAG and trafilatura_extract_func:
            logger.debug(f"Proceeding to trafilatura for {url}")
            content = trafilatura_extract_func(downloaded_html, include_comments=False, include_tables=False, output_format='txt', favor_precision=True, url=url)
            if content:
                content = "\n".join([line for line in content.splitlines() if line.strip()])
                logger.info(f"Trafilatura extracted for: {url}"); return content
            else: # Trafilatura failed, try BS
                logger.warning(f"Trafilatura failed for {url}. Trying BeautifulSoup.")
                # (Your BS logic here - keep it, or simplify for disabled mode)
                try:
                    soup = BeautifulSoup(downloaded_html, 'html.parser')
                    # Simplified BS fallback for this example
                    body_text = soup.body.get_text(separator='\n',strip=True) if soup.body else ""
                    body_text = "\n".join([line for line in body_text.splitlines() if line.strip()])
                    logger.info(f"BS body fallback for: {url}"); return body_text if body_text else None
                except Exception as e_bs: logger.error(f"BS fallback error for {url}: {e_bs}"); return None
        elif settings.DISABLE_RAG:
            logger.warning(f"RAG disabled, returning snippet for {url}")
            return downloaded_html[:500] + "... (extraction skipped)"
        else: # Trafilatura not loaded for some reason even if RAG not disabled
             logger.error(f"Trafilatura function not available for {url} despite RAG mode.")
             return downloaded_html[:500] + "... (extraction function missing)"
    else:
        logger.warning(f"Failed to download or HTML empty for: {url}")
    return None
    # --- End of your fetch_article_content (ensure all paths handled) ----

def chunk_text(text: str, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP) -> List[str]:
    if settings.DISABLE_RAG:
        logger.debug("Chunking called but RAG is disabled. Returning first part of text as single chunk.")
        return [text[:chunk_size]] if text else []
    # ... (Your existing chunk_text logic, ensuring get_text_tokenizer() is used) ...
    # This function is copied from your provided code
    logger.debug(f"chunk_text: Entered. Text length: {len(text)}, Chunk_size: {chunk_size}, Overlap: {chunk_overlap}")
    if not text or tokenizer is None: logger.warning("chunk_text: No text or tokenizer available."); return []
    try: tokens = get_text_tokenizer().encode(text)
    except Exception as e: logger.error(f"chunk_text: Failed to encode text: {e}", exc_info=True); return []
    if not tokens: logger.warning("chunk_text: No tokens from text."); return []
    chunks = []
    current_pos = 0; max_tokens = len(tokens)
    while current_pos < max_tokens:
        end_pos = min(current_pos + chunk_size, max_tokens)
        chunk_text_content = get_text_tokenizer().decode(tokens[current_pos:end_pos])
        if chunk_text_content.strip(): chunks.append(chunk_text_content.strip())
        if end_pos == max_tokens: break
        step = chunk_size - chunk_overlap
        if step <= 0: logger.warning("chunk_text: Step not positive. Advancing by 1."); step = 1
        current_pos += step
    logger.info(f"chunk_text: Original (tokens: {len(tokens)}) chunked into {len(chunks)} chunks.")
    return chunks

def ensure_qdrant_collection() -> bool:
    if settings.DISABLE_RAG:
        logger.info("ensure_qdrant_collection: RAG disabled, skipping Qdrant check.")
        return True # Assume OK for disabled mode
    if not qdrant_client: logger.error("ensure_qdrant_collection: Qdrant client not initialized."); return False
    if not embedding_model: logger.error("ensure_qdrant_collection: Embedding model not initialized."); return False
    # ... (Your existing ensure_qdrant_collection logic, ensure it uses qdrant_models.VectorParams etc.) ...
    # This function is copied from your provided code, with qdrant_models fix
    try:
        logger.debug(f"Checking Qdrant collection '{settings.VECTOR_DB_COLLECTION_NAME}'.")
        qdrant_client.get_collection(collection_name=settings.VECTOR_DB_COLLECTION_NAME)
        logger.info(f"Collection '{settings.VECTOR_DB_COLLECTION_NAME}' exists. Dim: {EMBEDDING_DIM}")
        return True
    except Exception as e:
        is_not_found = ("not found" in str(e).lower() or "collectionnotfoundexception" in str(e).lower().replace("_","") or (hasattr(e, 'status_code') and e.status_code == 404))
        if is_not_found:
            logger.info(f"Collection '{settings.VECTOR_DB_COLLECTION_NAME}' not found. Creating...")
            try:
                qdrant_client.recreate_collection(
                    collection_name=settings.VECTOR_DB_COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(size=EMBEDDING_DIM, distance=qdrant_models.Distance.COSINE)
                )
                logger.info(f"Collection created with dim {EMBEDDING_DIM}"); return True
            except Exception as ce: logger.exception(f"Error CREATING collection: {ce}"); return False
        else: logger.exception(f"Error CHECKING collection: {e}"); return False
    finally: logger.debug("ensure_qdrant_collection: Exiting.")


async def ingest_single_article(
    url: Optional[str]=None, text_content: Optional[str]=None,
    source_name: Optional[str]="Unknown", article_id_override: Optional[str]=None
) -> Dict:
    if settings.DISABLE_RAG:
        logger.warning(f"Ingest attempt for source '{source_name}' while RAG is disabled.")
        return {"status": "error", "message": "Ingestion and RAG features are disabled in assessment mode."}

    logger.info(f"ingest_single_article: Processing source='{source_name}', URL='{url if url else 'N/A'}'")
    if not all([qdrant_client, embedding_model, tokenizer, trafilatura_extract_func]): # Added trafilatura check
        logger.error("ingest_single_article: Core RAG components (Qdrant, Embedder, Tokenizer, Trafilatura) not initialized.")
        return {"status": "error", "message": "Core RAG components for ingestion not ready."}
    # ... (Your existing ingest_single_article logic) ...
    # This function is copied from your provided code
    if not ensure_qdrant_collection(): return {"status":"error", "message":"Qdrant collection error."}
    actual_content: Optional[str] = None
    if url and not text_content: actual_content = await fetch_article_content(url)
    elif text_content: actual_content = text_content
    if not (actual_content and actual_content.strip()):
        message = f"Content empty for URL: {url}" if url else "Content empty."
        logger.warning(f"ingest_single_article: {message}"); return {"status":"error", "message": message}
    article_chunks = chunk_text(actual_content)
    if not article_chunks: return {"status":"error", "message":"No chunks from content."}
    base_article_id_for_payload = article_id_override or (hashlib.sha256(url.encode('utf-8')).hexdigest() if url else str(uuid.uuid4()))
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
        except Exception as e: logger.error(f"Error processing chunk {i} for article_id {base_article_id_for_payload}: {e}", exc_info=True)
    if not points: return {"status":"warning", "message":f"No valid points for article_id {base_article_id_for_payload}."}
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
        logger.warning("INGESTION_SERVICE: Module imported. RAG features are DISABLED.")
    else:
        logger.info("INGESTION_SERVICE: Module imported. Verifying RAG component initializations...")
        if not all([qdrant_client, embedding_model, tokenizer, trafilatura_extract_func]):
            logger.critical("INGESTION_SERVICE: One or more RAG components failed to initialize properly during module load.")
        else:
            logger.info("INGESTION_SERVICE: All RAG components seem initialized. Ensuring Qdrant collection.")
            if not ensure_qdrant_collection(): # ensure_qdrant_collection itself logs errors
                logger.critical("INGESTION_SERVICE: Qdrant collection check/creation failed post-import.")
            else:
                logger.info("INGESTION_SERVICE: Qdrant collection is ready.")