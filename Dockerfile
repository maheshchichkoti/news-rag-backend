# news-rag-backend/Dockerfile
FROM python:3.10-slim-buster

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 # Port the application inside the container will listen on

WORKDIR /app

# For Render deployment (assessment mode), use requirements-minimal.txt
COPY requirements-minimal.txt ./requirements.txt
# For local full build, comment above line and uncomment below line:
# COPY requirements-full.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # If using requirements-full.txt, you would add:
    # --extra-index-url https://download.pytorch.org/whl/cpu \
    rm -rf /root/.cache

COPY ./app ./app

EXPOSE 8000

# CMD for Gunicorn with 1 worker, suitable for Render free tier.
# Render's "Start Command" can override this if needed.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:${PORT}", "app.main:app"]