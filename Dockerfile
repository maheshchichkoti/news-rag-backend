# news-rag-backend/Dockerfile

# ARG DOCKER_REQUIREMENTS_FILE=requirements-minimal.txt # Default to minimal
# Use python:3.10-slim-buster. Alpine can be tricky with C extensions.
FROM python:3.10-slim-buster

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # PORT will be set by Render or defaults in CMD. Let Gunicorn read it.
    # GUNICORN_WORKERS and STARTUP_DELAY are better as Render Env Vars if possible,
    # or directly in the CMD if Render doesn't easily pass them to the exec bash -c.
    # For simplicity in CMD, we'll use fixed values or rely on Render env vars.
    PORT=8000 # Default port, Render will map to this

WORKDIR /app

# System dependencies for some Python packages (if minimal reqs ever grow to need them)
# For truly minimal, this might not even be needed if requirements-minimal.txt is pure Python.
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends gcc python3-dev build-essential && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# For Render deployment, we'll use requirements-minimal.txt
COPY requirements-minimal.txt ./requirements.txt
# For local full build, you'd comment above and uncomment below, or use build-args
# COPY requirements-full.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # If using requirements-full.txt, add --extra-index-url https://download.pytorch.org/whl/cpu
    # Example for full:
    # pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt && \
    rm -rf /root/.cache /tmp/*

COPY ./app ./app
# Scripts (like ingest_news.py) are not copied as they are not needed for API runtime

EXPOSE 8000 # Port the application WILL listen on

# CMD for Gunicorn with 1 worker for free tier
# Render's "Start Command" can override this.
# If Render uses this CMD, ensure it has access to PORT, GUNICORN_WORKERS, STARTUP_DELAY env vars.
# Simpler CMD for Render free tier focusing on stability:
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:8000", "app.main:app"]

# Your more complex CMD with startup delay (ensure Render sets STARTUP_DELAY and GUNICORN_WORKERS env vars):
# CMD exec bash -c "echo 'Delaying startup for ${STARTUP_DELAY:-45}s...' && \
#     sleep ${STARTUP_DELAY:-45} && \
#     echo 'Starting Gunicorn server...' && \
#     gunicorn -k uvicorn.workers.UvicornWorker \
#     -w ${GUNICORN_WORKERS:-1} \
#     -b 0.0.0.0:${PORT:-8000} \
#     --timeout 120 \
#     --preload \
#     app.main:app"