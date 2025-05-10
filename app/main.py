# app/main_barebones.py
import logging # Keep logging to see output
import os

from fastapi import FastAPI

# Minimal logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("Starting BAREBONES FastAPI app...")
# If your original main.py imported settings at the top,
# you might need a dummy settings object or to remove that import for this test.
# For now, assume we don't need to import your full settings.py for this test.

app = FastAPI(title="Barebones Test API")

@app.on_event("startup")
async def startup_event():
    logger.info("BAREBONES app startup complete.")
    # Attempt to read PORT env var to confirm it's passed
    port_env = os.getenv("PORT", "NOT_SET")
    logger.info(f"PORT environment variable is: {port_env}")


@app.get("/health")
async def health_check():
    logger.info("Barebones /health endpoint hit")
    return {"status": "ok", "message": "Barebones app is alive!"}

@app.get("/")
async def root():
    logger.info("Barebones / (root) endpoint hit")
    return {"message": "Welcome to the Barebones Test API"}

# No other imports, no other endpoints, no service calls.
logger.info("BAREBONES FastAPI app configured.")