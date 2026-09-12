"""
VoiceGuard FastAPI Application Entrypoint
Problem Statement SIH26104: AI-Powered Real-Time Detection & Prevention of Voice Cloning Impersonation Attacks
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.database import engine, Base
from backend.app.api.calls import router as calls_router
from backend.app.api.profiles import router as profiles_router
from backend.app.api.transactions import router as transactions_router
from backend.app.api.websocket import router as websocket_router

# Ensure database tables exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Probabilistic Voice Integrity & Financial Fraud Prevention Gateway for SIH26104."
)

# CORS Middleware (Enable for web dashboards and local tunnels)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(calls_router, prefix=settings.API_V1_PREFIX)
app.include_router(profiles_router, prefix=settings.API_V1_PREFIX)
app.include_router(transactions_router, prefix=settings.API_V1_PREFIX)
app.include_router(websocket_router)

# Health Check
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ONLINE",
        "gateway": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "raw_audio_retention": settings.RAW_AUDIO_RETENTION,
        "model_loaded": True
    }

# Serve Frontend static assets if directory exists
frontend_dir = os.path.abspath(r"D:\voiceguard\frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend_root():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "VoiceGuard Gateway API is running. Frontend index.html not found."}
    @app.get("/overview", include_in_schema=False)
    @app.get("/screens/overview.html", include_in_schema=False)
    def serve_overview():
        return FileResponse(os.path.join(frontend_dir, "screens", "overview.html"))

    @app.get("/live-inspection", include_in_schema=False)
    @app.get("/screens/live_inspection.html", include_in_schema=False)
    def serve_live_inspection():
        return FileResponse(os.path.join(frontend_dir, "screens", "live_inspection.html"))

    @app.get("/analytics", include_in_schema=False)
    @app.get("/screens/analytics.html", include_in_schema=False)
    def serve_analytics():
        return FileResponse(os.path.join(frontend_dir, "screens", "analytics.html"))
