# Use official Python slim image
FROM python:3.10-slim-buster

# Set environment variables (NO indentation before ENV)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=1 \
    STARTUP_DELAY=30 \
    MEMORY_LIMIT_MB=450

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# Copy application code
COPY . .

# Memory-optimized startup command
CMD exec bash -c "echo 'Delaying startup for ${STARTUP_DELAY}s...' && \
    sleep ${STARTUP_DELAY} && \
    echo 'Starting memory-limited server (${MEMORY_LIMIT_MB}MB max)' && \
    gunicorn -k uvicorn.workers.UvicornWorker \
    -w ${GUNICORN_WORKERS} \
    --threads ${GUNICORN_THREADS} \
    -b 0.0.0.0:${PORT} \
    --timeout 120 \
    --preload \
    --max-requests 50 \
    --max-requests-jitter 10 \
    app.main:app"