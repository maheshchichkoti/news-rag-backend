# test_ingestion.py
import asyncio
import sys
import os
import traceback # For printing full tracebacks

print("test_ingestion.py: Script started. Attempting imports...")
try:
    from app.services import ingestion_service
    from app.core.config import settings
    print("test_ingestion.py: Imports successful.")
except ImportError as e:
    print(f"test_ingestion.py: ImportError: {e}. Check PYTHONPATH or if script is run from project root with venv active.")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"test_ingestion.py: Unexpected error during imports: {e}")
    traceback.print_exc()
    sys.exit(1)


async def main():
    print("test_ingestion.py: Async main() started.")

    if not ingestion_service.qdrant_client or \
       not ingestion_service.embedding_model or \
       not ingestion_service.tokenizer:
        print("Test script: One or more core services (Qdrant, Embedder, Tokenizer) from ingestion_service are not initialized. Exiting.")
        return

    print(f"Test script: Using Qdrant collection from settings: {settings.VECTOR_DB_COLLECTION_NAME}")
    print("Test script: Explicitly ensuring Qdrant collection from test script...")
    if not ingestion_service.ensure_qdrant_collection():
        print("Test script: Failed to ensure Qdrant collection via explicit call. Exiting.")
        return
    print(f"Test script: Qdrant collection '{settings.VECTOR_DB_COLLECTION_NAME}' is ready (verified by explicit call).")

    print(f"Test script: Embedding dimension from service: {ingestion_service.EMBEDDING_DIM}")

    # test_url = "https://apnews.com/article/israel-palestinians-gaza-hamas-war-news-04-18-2024-a7a6d7a0911f310065e37792878bf71b"
    test_url = "http://example.com" # <<< CHANGE THIS LINE

    print(f"\nTest script: Attempting to ingest: {test_url}")
    result = await ingestion_service.ingest_single_article(url=test_url, source_name="AP News Test")
    print(f"Test script: Ingestion result for {test_url}: {result}")

    if result.get("status") == "success":
        print("Test script: INGESTION SUCCEEDED. Check Qdrant dashboard.")
    else:
        print("Test script: INGESTION FAILED or had warnings.")

    print("test_ingestion.py: Async main() finished.")


if __name__ == "__main__":
    print("test_ingestion.py: __name__ == '__main__'. Running asyncio.run(main()).")
    try:
        asyncio.run(main())
    except Exception as e_main_run:
        print(f"test_ingestion.py: Unhandled exception in asyncio.run(main): {e_main_run}")
        traceback.print_exc()
    finally:
        print("test_ingestion.py: Script execution complete.")