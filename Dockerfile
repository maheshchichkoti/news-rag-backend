# news-rag-backend/Dockerfile

# 1. Base Image: Use an official Python image.
FROM python:3.10-slim

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# 3. Set Work Directory
WORKDIR /app

# 4. Copy requirements first to leverage Docker cache
COPY requirements.txt .

# 5. Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code
COPY ./app ./app
# COPY ./scripts ./scripts # Kept commented out

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Command to run the application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]