FROM python:3.10-slim-buster

# Absolute minimum environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    GUNICORN_WORKERS=1 \
    STARTUP_DELAY=45

# Lightweight dependency installation
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Install requirements with strict memory control
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    find /usr/local/lib/python3.10 -name '__pycache__' -exec rm -rf {} + && \
    rm -rf /root/.cache

COPY . .

# Ultra-lightweight startup command
CMD exec bash -c "echo 'Delaying startup for ${STARTUP_DELAY}s...' && \
    sleep ${STARTUP_DELAY} && \
    echo 'Starting single-worker server' && \
    gunicorn -k uvicorn.workers.UvicornWorker \
    -w ${GUNICORN_WORKERS} \
    -b 0.0.0.0:${PORT} \
    --timeout 120 \
    --preload \
    --max-requests 30 \
    app.main:app"