# 1. Use slim-buster for better compatibility
FROM python:3.10-slim-buster

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app

# 3. Install system dependencies first (required for some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 4. Set Work Directory
WORKDIR /app

# 5. Copy requirements first to leverage Docker cache
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# 7. Copy application code
COPY ./app ./app

# 8. Reduce Gunicorn workers for Render's free tier (1GB RAM)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "--threads", "2", "-b", "0.0.0.0:$PORT", "--timeout", "120", "app.main:app"]