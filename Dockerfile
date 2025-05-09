# news-rag-backend/Dockerfile

# 1. Base Image: Use an official Python image.
# Choose a version that matches your local development (e.g., 3.10, 3.11).
# Slim versions are smaller.
FROM python:3.10-slim

# 2. Set Environment Variables (Optional but good practice)
ENV PYTHONDONTWRITEBYTECODE 1  # Prevents python from writing .pyc files to disc
ENV PYTHONUNBUFFERED 1       # Prevents python from buffering stdout/stderr

# 3. Set Work Directory
WORKDIR /app

# 4. Copy requirements first to leverage Docker cache
COPY requirements.txt .

# 5. Install Dependencies
# --no-cache-dir reduces image size
# Make sure your requirements.txt is clean and only has necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code
COPY ./app ./app
# COPY ./scripts ./scripts # If any scripts are needed at runtime (unlikely for API) <-- COMMENTED OUT
# If you have other directories like a 'data' dir for static assets needed by the app, copy them too.

# 7. Expose the port the app runs on (matches Uvicorn command)
EXPOSE 8000

# 8. Command to run the application
# This will be overridden by Render.com's start command, but good for local Docker testing.
# Use Gunicorn as a production-ready ASGI server with Uvicorn workers.
# You'll need to add gunicorn to your requirements.txt
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]