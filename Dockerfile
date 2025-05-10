# news-rag-backend/Dockerfile
FROM python:3.10-slim-buster

# Environment variables (no inline comments!)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# For Render deployment (assessment mode)
COPY requirements-minimal.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache

COPY ./app ./app

EXPOSE 8000

# Simple Gunicorn command
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:8000", "app.main:app"]