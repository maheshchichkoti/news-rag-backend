# 1. Use slim-buster for better compatibility
FROM python:3.10-slim-buster

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=2 \
    STARTUP_DELAY=20  # Increased delay for RAG services

# 3. Install system dependencies first (required for some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 4. Set work directory
WORKDIR /app

# 5. Copy requirements first to leverage Docker cache
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# 7. Copy application code
COPY . .

# 8. Use shell form CMD to properly expand $PORT
CMD exec bash -c "echo 'Waiting ${STARTUP_DELAY}s for services to initialize...' && \
    sleep ${STARTUP_DELAY} && \
    echo 'Starting Gunicorn with ${GUNICORN_WORKERS} worker(s) and ${GUNICORN_THREADS} threads...' && \
    gunicorn -k uvicorn.workers.UvicornWorker \
    -w ${GUNICORN_WORKERS} \
    --threads ${GUNICORN_THREADS} \
    -b 0.0.0.0:${PORT} \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    app.main:app"