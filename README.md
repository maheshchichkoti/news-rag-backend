# News RAG Chatbot - Backend

## Overview

This FastAPI backend powers a RAG (Retrieval-Augmented Generation) chatbot for querying news articles. It uses Jina Embeddings, Qdrant as a vector store, Google Gemini for generation, and Redis for session management.

## Tech Stack

- Python 3.10+
- FastAPI
- Uvicorn (with Gunicorn for production)
- Qdrant (Vector Database)
- Jina Embeddings (via `sentence-transformers`)
- Google Gemini API (via `google-generativeai`)
- Redis (for chat history)
- Docker

## Local Development Setup

1.  **Clone the repository:**

    ```bash
    git clone <your-backend-repo-url>
    cd news-rag-backend
    ```

2.  **Create and activate a Python virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows Git Bash/Linux/macOS
    # venv\Scripts\activate  # On Windows CMD/PowerShell
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the project root (`news-rag-backend/.env`) from the `.env.example`:

    ```env
    # .env.example
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_FROM_GCP_OR_AI_STUDIO"
    QDRANT_URL="http://localhost:6333"
    QDRANT_API_KEY="" # Optional: for local Qdrant if not secured
    REDIS_URL="redis://localhost:6379/0"

    # Model Names (defaults are usually fine, but configurable)
    JINA_HF_MODEL_NAME="jinaai/jina-embeddings-v2-base-en"
    GEMINI_MODEL_NAME="models/gemini-1.5-flash-latest" # Or your preferred Gemini model
    VECTOR_DB_COLLECTION_NAME="news_articles_v2"
    ```

    Fill in your `GOOGLE_API_KEY`.

5.  **Run Local Services (Qdrant & Redis):**
    Ensure Docker Desktop is running.

    ```bash
    docker compose up -d
    ```

    This will start Qdrant on port 6333 and Redis on port 6379.

6.  **Run the FastAPI Application:**
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```
    The API will be available at `http://localhost:8000`. Access Swagger docs at `http://localhost:8000/docs`.

## Ingestion

Article content is ingested from local `.txt` files.

1.  Place article text files in `data/raw_articles/`.
2.  (Optional) Add corresponding metadata to `data/metadata.json`.
3.  Run the ingestion script:
    `bash
python scripts/ingest_news.py
`
    This approach was chosen to focus on the RAG pipeline rather than complex web scraping for this assignment. For production, a robust scraping/fetching mechanism (e.g., using Playwright or dedicated scraping services) would be needed.

## Deployment (Example: Render.com)

1.  **Containerize the Application:**
    This project includes a `Dockerfile` to build an image for the FastAPI application. Ensure `gunicorn` is in `requirements.txt`.

2.  **Set up External Services:**

    - **Qdrant:** Use Qdrant Cloud (free tier available) or deploy Qdrant as a Docker container on your hosting platform.
    - **Redis:** Use a managed Redis service (e.g., Redis Cloud, Upstash, or an addon from your hosting provider).

3.  **Configure Environment Variables on Hosting Platform:**
    Set the following environment variables in your Render.com service (or equivalent):

    - `GOOGLE_API_KEY`: Your Gemini API key.
    - `QDRANT_URL`: The URL of your deployed Qdrant instance (e.g., from Qdrant Cloud).
    - `QDRANT_API_KEY`: The API key for your deployed Qdrant instance (if applicable).
    - `REDIS_URL`: The connection URL for your deployed Redis instance.
    - `PYTHON_VERSION`: (If required by platform) e.g., `3.10.12`
    - (Optional) `JINA_HF_MODEL_NAME`, `GEMINI_MODEL_NAME`, `VECTOR_DB_COLLECTION_NAME` if you need to override defaults.

4.  **Set Start Command (Render.com example):**
    Render will typically detect the `Dockerfile`. If you need to specify a start command for a Web Service:

    ```bash
    gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:$PORT app.main:app
    ```

    (`$PORT` is usually provided by the hosting platform).

5.  **Ingest Data into Deployed Qdrant:**
    - After deploying Qdrant, you'll need to populate it. The simplest way is to run the `scripts/ingest_news.py` script from your local machine, but configure your local `.env` temporarily to point `QDRANT_URL` and `QDRANT_API_KEY` to your _cloud_ Qdrant instance.
    - Alternatively, for a small demo set, you could create a protected API endpoint to trigger ingestion for a few articles.

## API Endpoints

_(List your main endpoints as described previously)_

- `POST /session/new`
- `POST /chat` (Headers: `X-Session-Id`)
- `GET /chat/history/{session_id}`
- `POST /chat/session/{session_id}/clear`
- `POST /ingest` (for local/manual ingestion if extended)
- `GET /health`

## RAG Pipeline & Caching

_(Briefly describe the RAG flow and Redis usage as planned for your code walkthrough)_

- **Chat History:** Stored in Redis as lists per `session_id`. Currently, history is kept for `MAX_HISTORY_LENGTH` (10 turns). TTLs could be added via `redis_client.expire(key, <seconds>)` if sessions should auto-expire.
- **LLM Call Caching (Future Improvement):** For identical (or semantically similar) queries with the same context, responses could be cached in Redis to reduce LLM calls. This is not implemented but is a standard optimization. Cache warming is not applicable at this project's current scale for LLM responses.

## Potential Future Improvements

- Robust, automated news fetching and processing pipeline.
- Advanced context re-ranking before sending to LLM.
- LLM response caching.
- User authentication.
- More detailed observability (logging, tracing, metrics).
