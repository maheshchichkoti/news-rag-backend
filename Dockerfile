FROM python:3.10-slim-buster

# Set environment variables (ensure no trailing spaces after backslashes)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=1 \
    STARTUP_DELAY=15

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

# Startup command
CMD exec bash -c "echo 'Delaying startup for ${STARTUP_DELAY}s...' && \
    sleep ${STARTUP_DELAY} && \
    echo 'Starting server with ${GUNICORN_WORKERS} worker and ${GUNICORN_THREADS} thread' && \
    gunicorn -k uvicorn.workers.UvicornWorker \
    -w ${GUNICORN_WORKERS} \
    --threads ${GUNICORN_THREADS} \
    -b 0.0.0.0:${PORT} \
    --timeout 120 \
    --preload \
    app.main:app"