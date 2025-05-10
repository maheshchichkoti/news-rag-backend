# app/core/config_barebones.py
# Minimal stub to allow main_barebones.py to import 'settings' if it must.
import os
print("INFO: [config_barebones.py] Loaded barebones config.")

class BarebonesSettings:
    # Add any settings that app/main_barebones.py *absolutely must have* at import time
    # For the barebones main, it probably needs nothing.
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # We don't need DISABLE_RAG here as nothing RAG related is loaded

settings = BarebonesSettings()