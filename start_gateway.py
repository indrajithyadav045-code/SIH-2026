"""
VoiceGuard Gateway Local Runner
Starts Uvicorn server on port 8000 hosting the FastAPI backend and Frontend Command Center.
"""

import sys
import os
import uvicorn

# Safe console encoding
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    print("=" * 75)
    print("Launching VoiceGuard Gateway (SIH26104) on http://127.0.0.1:8000")
    print("Frontend UI: http://127.0.0.1:8000/")
    print("API Docs:    http://127.0.0.1:8000/docs")
    print("=" * 75)
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, log_level="info")