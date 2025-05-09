FROM python:3.10-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=2 \
    STARTUP_DELAY=20  # Increased delay for RAG services

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

COPY . .

# Use shell form for proper variable expansion and startup sequencing
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