FROM python:3.10-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=1 \  # Reduced to 1 thread
    STARTUP_DELAY=15

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies with cache cleanup
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

COPY . .

# Optimized startup command
CMD exec bash -c "echo 'Delaying startup for ${STARTUP_DELAY}s...' && \
    sleep ${STARTUP_DELAY} && \
    echo 'Starting server with ${GUNICORN_WORKERS} worker and ${GUNICORN_THREADS} thread' && \
    gunicorn -k uvicorn.workers.UvicornWorker \
    -w ${GUNICORN_WORKERS} \
    --threads ${GUNICORN_THREADS} \
    -b 0.0.0.0:${PORT} \
    --timeout 120 \
    --preload \  # Important for memory optimization
    app.main:app"