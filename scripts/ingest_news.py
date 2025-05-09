# scripts/ingest_news.py
import asyncio
import os
import sys
import json

# --- Add project root to sys.path ---
# This assumes the script is in a 'scripts' subdirectory of the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
print(f"Added to sys.path: {PROJECT_ROOT}") # For debugging, can remove later
# --- End of path modification ---

from app.services import ingestion_service # Your existing service
from app.core.config import settings # If needed for collection name, etc.

# Adjust path to your raw articles and metadata
# Assuming this script is run from the `news-rag-backend` root.
ARTICLES_DIR = "data/raw_articles"
METADATA_FILE = "data/metadata.json" # Optional

async def ingest_local_articles():
    print("Starting ingestion of local articles...")

    # Ensure Qdrant collection exists (important if running script standalone)
    # The ingestion_service module tries to do this on import, but an explicit check here is good.
    if not ingestion_service.qdrant_client or not ingestion_service.embedding_model:
        print("CRITICAL: Qdrant client or embedding model not initialized in ingestion_service. Cannot proceed.")
        return
    
    print("Ensuring Qdrant collection from script...")
    if not ingestion_service.ensure_qdrant_collection():
        print("CRITICAL: Failed to ensure Qdrant collection. Aborting.")
        return
    print(f"Qdrant collection '{settings.VECTOR_DB_COLLECTION_NAME}' is ready.")

    # Option 1: Simple iteration over .txt files in a directory
    if not os.path.exists(ARTICLES_DIR):
        print(f"Articles directory not found: {ARTICLES_DIR}")
        return

    article_files = [f for f in os.listdir(ARTICLES_DIR) if f.endswith(".txt")]
    print(f"Found {len(article_files)} .txt files in {ARTICLES_DIR}.")

    # Load metadata if available
    metadata_map = {}
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            for item in metadata_list:
                metadata_map[item["filename"]] = {"url": item.get("original_url"), "source": item.get("source_name", "Unknown")}
            print(f"Loaded metadata for {len(metadata_map)} articles.")
        except Exception as e:
            print(f"Warning: Could not load or parse metadata file {METADATA_FILE}: {e}")
    
    success_count = 0
    fail_count = 0

    for i, filename in enumerate(article_files):
        filepath = os.path.join(ARTICLES_DIR, filename)
        print(f"\n--- Processing article {i+1}/{len(article_files)}: {filename} ---")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text_content = f.read()

            if not text_content.strip():
                print(f"Skipping empty file: {filename}")
                fail_count +=1
                continue

            # Get metadata or use defaults
            meta = metadata_map.get(filename, {})
            source_url = meta.get("url", f"localfile://{filename}") # Use a placeholder URL
            source_name = meta.get("source", "Local File")
            
            # Use filename (without .txt) or a hash as article_id_override for consistency
            article_id = os.path.splitext(filename)[0] 

            result = await ingestion_service.ingest_single_article(
                text_content=text_content,
                source_name=source_name,
                url=source_url, # Pass the original URL if you have it
                article_id_override=article_id # Use a consistent ID based on filename
            )
            
            if result.get("status") == "success":
                print(f"Successfully ingested: {filename} (Article ID: {result.get('article_id')})")
                success_count += 1
            else:
                print(f"Failed to ingest: {filename}. Reason: {result.get('message')}")
                fail_count += 1

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
        
        # Optional: Add a small delay if you're processing many and want to be gentle
        # await asyncio.sleep(0.1) 

    print(f"\n--- Ingestion Summary ---")
    print(f"Successfully ingested: {success_count} articles.")
    print(f"Failed to ingest: {fail_count} articles.")

if __name__ == "__main__":
    print("Running ingest_news.py script...")
    # Make sure venv is active and Docker services (Qdrant/Redis) are running
    asyncio.run(ingest_local_articles())
    print("Ingestion script finished.")