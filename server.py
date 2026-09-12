"""
VoiceGuard Full-Stack FastAPI Backend Server
Problem Statement SIH26104: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

Serves:
- Interactive Web Application with Three.js 3D Acoustic Shield
- Real-time Audio Ingestion (Microphone, File Upload, Ingress Streams)
- Model Inference via VoiceGuard Wav2Vec 2.0 with Partial Unfreezing
- Forensic Acoustic Telemetry (SNR, ZCR, Spectral Centroid, Flatness, Anomaly Tags)
- Automated Kill-Switch Trigger & Audit Logging
"""

import os
import io
import time
import json
import base64
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from voiceguard_wav2vec2 import VoiceGuardWav2Vec2, extract_forensic_metrics, LossyAudioAugmenter
from inference import VoiceGuardInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("VoiceGuard.Server")

app = FastAPI(
    title="VoiceGuard Voice Cloning Defense API",
    description="SIH26104 Production Inference & Real-Time Biometrics Telemetry",
    version="2.0.0"
)

# Enable CORS for institutional IVR and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize VoiceGuard Engine
BACKBONE_PATH = "D:/ML model VG" if os.path.isdir("D:/ML model VG") else "facebook/wav2vec2-base"
CHECKPOINT_PATH = "D:/voiceguard/checkpoints/best_voiceguard_wav2vec2.pt"

logger.info("Initializing VoiceGuard Inference Engine...")
engine = VoiceGuardInferenceEngine(
    checkpoint_path=CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None,
    backbone_model_name=BACKBONE_PATH,
    threshold=0.75,
    temperature=1.0
)
logger.info("VoiceGuard Engine Ready.")

# In-memory audit log store
AUDIT_LOGS: List[Dict[str, Any]] = []

def record_log(data: Dict[str, Any], filename: str = "Live Audio Stream"):
    """Appends an event to the forensic audit log."""
    log_entry = {
        "id": f"VG-{int(time.time()*1000)%1000000:06d}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "decision": data.get("decision", "UNKNOWN"),
        "threat_score": data.get("threat_score", 0.0),
        "threat_level": data.get("threat_level", "UNKNOWN"),
        "vocoder": data.get("vocoder_analysis", {}).get("predicted_subtype", "unknown"),
        "vocoder_confidence": data.get("vocoder_analysis", {}).get("confidence", 0.0),
        "anomaly_tags": data.get("forensic_signal_metrics", {}).get("anomaly_tags", []),
        "latency_ms": data.get("latency_ms", 0.0),
        "snr_db": data.get("forensic_signal_metrics", {}).get("snr_db", 0.0)
    }
    AUDIT_LOGS.insert(0, log_entry)
    # Keep last 100 entries
    if len(AUDIT_LOGS) > 100:
        AUDIT_LOGS.pop()
    return log_entry


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main VoiceGuard web dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>VoiceGuard Backend Running. static/index.html not found.</h2>")


@app.post("/api/analyze-audio")
async def analyze_audio(
    file: UploadFile = File(...),
    threshold: float = Form(0.75),
    temperature: float = Form(1.0)
):
    """
    Analyzes uploaded audio file (.wav, .mp3, .ogg, .webm, etc.)
    Runs through fine-tuned Wav2Vec 2.0 with partial unfreezing and returns forensic telemetry.
    """
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Read audio in memory using soundfile
        bio = io.BytesIO(content)
        try:
            audio_data, sr = sf.read(bio)
        except Exception as e:
            # Fallback: try torchaudio
            bio.seek(0)
            import torchaudio
            waveform, sr = torchaudio.load(bio)
            audio_data = waveform.squeeze().cpu().numpy()

        # Update engine threshold & temperature dynamically
        engine.threshold = threshold
        engine.temperature = temperature

        # Inspect through VoiceGuard model
        result = engine.inspect_audio(audio_data, sr=sr, return_embedding=False)
        log_entry = record_log(result, filename=file.filename or "Uploaded Sample")
        result["log_id"] = log_entry["id"]

        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate-attack")
async def simulate_attack(
    attack_type: str = Form("diffwave"),
    threshold: float = Form(0.75)
):
    """
    Simulates real-world neural vocoder cloning attacks on the fly.
    Generates realistic speech-like waveform with synthetic artifacts:
    - 'diffwave': Phase jitter & high-frequency diffusion artifacts
    - 'elevenlabs': Unnatural pitch flatness & prosodic discontinuities
    - 'melgan': Spectral inversion & buzzy harmonic glints
    - 'authentic': Pristine biological vocal tract resonance
    """
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = random.uniform(140.0, 220.0)

    # Base biological vocal tract formant harmonic structure
    signal = (
        0.5 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.15 * np.sin(2 * np.pi * 3 * f0 * t) +
        0.05 * np.sin(2 * np.pi * 4 * f0 * t)
    )

    is_spoof = (attack_type.lower() != "authentic")

    if attack_type == "diffwave":
        # Latent phase jitter
        jitter = np.random.normal(0, 0.08, len(t))
        signal += 0.35 * np.sin(2 * np.pi * 3800.0 * (t + jitter))
        signal += 0.05 * np.random.normal(0, 0.1, len(t))
    elif attack_type == "elevenlabs":
        # Flat pitch contour + phase modulation
        signal = 0.6 * np.sin(2 * np.pi * 175.0 * t) + 0.3 * np.sin(2 * np.pi * 350.0 * t)
        signal += 0.15 * np.sin(2 * np.pi * 5200.0 * t)
    elif attack_type == "melgan":
        # High-frequency buzzy harmonics
        signal += 0.25 * np.sin(2 * np.pi * 4400.0 * t) + 0.15 * np.sin(2 * np.pi * 6600.0 * t)
    else:
        # Authentic biological subtle micro-tremor
        tremor = 0.02 * np.sin(2 * np.pi * 5.0 * t)
        signal *= (1.0 + tremor)
        signal += 0.01 * np.random.normal(0, 0.02, len(t))

    # Normalize to [-0.95, 0.95]
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.90
    signal = signal.astype(np.float32)

    # Run inference
    engine.threshold = threshold
    result = engine.inspect_audio(signal, sr=sr, return_embedding=False)
    
    # Generate WAV in-memory for immediate client playback
    bio = io.BytesIO()
    sf.write(bio, signal, sr, format="WAV")
    bio.seek(0)
    audio_b64 = base64.b64encode(bio.read()).decode("utf-8")

    result["audio_b64"] = f"data:audio/wav;base64,{audio_b64}"
    filename = f"Simulated {attack_type.upper()} Attack" if is_spoof else "Simulated Authentic Voice"
    log_entry = record_log(result, filename=filename)
    result["log_id"] = log_entry["id"]

    return JSONResponse(result)


@app.get("/api/sample-audio/{sample_type}")
async def get_sample_audio(sample_type: str):
    """Provides downloadable/playable sample audio clips."""
    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    if sample_type == "authentic":
        f0 = 160.0
        signal = 0.6 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
    else:
        f0 = 175.0
        jitter = np.random.normal(0, 0.08, len(t))
        signal = 0.6 * np.sin(2 * np.pi * f0 * (t + jitter)) + 0.3 * np.sin(2 * np.pi * 3800.0 * t)
        
    signal = (signal / (np.max(np.abs(signal)) + 1e-6) * 0.9).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, signal, sr, format="WAV")
    bio.seek(0)
    return Response(content=bio.read(), media_type="audio/wav")


@app.get("/api/system-status")
async def system_status():
    """Returns real-time model parameter breakdown and device telemetry."""
    total_params = sum(p.numel() for p in engine.model.parameters())
    trainable_params = sum(p.numel() for p in engine.model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return JSONResponse({
        "status": "OPERATIONAL",
        "problem_statement": "SIH26104",
        "architecture": "VoiceGuard Wav2Vec 2.0 (Top-3 Layers Unfrozen + Statistical Pooling)",
        "parameters": {
            "total": total_params,
            "frozen": frozen_params,
            "trainable": trainable_params,
            "frozen_pct": round(frozen_params / total_params * 100, 2),
            "trainable_pct": round(trainable_params / total_params * 100, 2)
        },
        "device": str(engine.device),
        "backbone": engine.model.pretrained_model_name,
        "default_threshold": engine.threshold,
        "active_defense_mode": "REAL_TIME_KILL_SWITCH",
        "audited_sessions_count": len(AUDIT_LOGS)
    })


@app.get("/api/forensic-logs")
async def get_forensic_logs():
    """Returns historical forensic inspection logs."""
    return JSONResponse(AUDIT_LOGS)


@app.post("/api/clear-logs")
async def clear_logs():
    """Clears audit logs."""
    AUDIT_LOGS.clear()
    return JSONResponse({"status": "CLEARED"})


if __name__ == "__main__":
    import uvicorn
    print("Starting VoiceGuard Full-Stack Server at http://127.0.0.1:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)