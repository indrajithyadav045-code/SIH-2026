"""
VoiceGuard Real-Time WebSocket Streaming Gateway
SIH26104: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

Provides low-latency binary audio chunk ingestion, rolling window inference,
and real-time telemetry push to connected security dashboards.
"""

import json
import base64
import logging
from typing import Dict, Set, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from backend.app.database import SessionLocal
from backend.app.models.schema_models import Call, User, VoiceProfile, RiskEvent
from backend.app.audio.preprocessor import preprocess_audio
from backend.app.audio.stream_buffer import buffer_registry
from backend.app.ml.synthetic_detector import get_synthetic_detector
from backend.app.ml.speaker_verifier import SpeakerVerifier
from backend.app.ml.liveness_detector import LivenessDetector
from backend.app.risk.behavior_analyzer import BehaviorAnalyzer
from backend.app.risk.context_engine import ContextEngine
from backend.app.risk.risk_engine import RiskEngine

logger = logging.getLogger("VoiceGuard.WebSocket")
router = APIRouter(tags=["WebSocket"])

synthetic_detector = get_synthetic_detector()
speaker_verifier = SpeakerVerifier()
liveness_detector = LivenessDetector()
behavior_analyzer = BehaviorAnalyzer()
context_engine = ContextEngine()
risk_engine = RiskEngine()


class ConnectionManager:
    """Manages active WebSocket telemetry connections keyed by call_id."""
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, call_id: str, websocket: WebSocket):
        await websocket.accept()
        if call_id not in self.active_connections:
            self.active_connections[call_id] = set()
        self.active_connections[call_id].add(websocket)
        logger.info(f"Dashboard client connected to telemetry stream: {call_id}")

    def disconnect(self, call_id: str, websocket: WebSocket):
        if call_id in self.active_connections:
            self.active_connections[call_id].discard(websocket)
            if not self.active_connections[call_id]:
                del self.active_connections[call_id]
        logger.info(f"Dashboard client disconnected from telemetry: {call_id}")

    async def broadcast_telemetry(self, call_id: str, message: dict):
        if call_id in self.active_connections:
            dead_sockets = set()
            for connection in self.active_connections[call_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception:
                    dead_sockets.add(connection)
            for dead in dead_sockets:
                self.active_connections[call_id].discard(dead)


manager = ConnectionManager()


@router.websocket("/ws/calls/{call_id}/telemetry")
async def websocket_call_telemetry(websocket: WebSocket, call_id: str):
    """Subscribes the frontend cybersecurity dashboard to real-time risk scores."""
    await manager.connect(call_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except (WebSocketDisconnect, RuntimeError):
        manager.disconnect(call_id, websocket)


@router.websocket("/ws/calls/{call_id}/audio")
@router.websocket("/ws/analyze")
async def websocket_audio_stream(websocket: WebSocket, call_id: Optional[str] = "LIVE-MIC"):
    """
    Real-Time Audio Ingestion WebSocket (Step 17):
    1. Receives raw audio chunks (binary Float32/Int16/WAV/WebM or JSON base64)
    2. Decodes via common preprocess_audio
    3. Appends to rolling 48,000-sample buffer (3.0s window, 1.0s step)
    4. Evaluates VAD / silence
    5. Runs Wav2Vec2 synthetic detection + speaker verification + risk aggregation
    6. Returns real-time JSON inference payload
    """
    await websocket.accept()
    buf = buffer_registry.get_or_create(call_id)
    logger.info(f"Live microphone WebSocket connected for session: {call_id}")

    db: Session = SessionLocal()
    try:
        call = db.query(Call).filter(Call.id == call_id).first()
        registered_phone = None
        vp_embedding = None
        if call and call.claimed_identity:
            user = db.query(User).filter((User.id == call.claimed_identity) | (User.name == call.claimed_identity)).first()
            if user:
                registered_phone = user.registered_phone
                vp = db.query(VoiceProfile).filter(VoiceProfile.user_id == user.id).first()
                if vp:
                    vp_embedding = vp.embedding

        while True:
            try:
                message = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                break

            if message.get("type") == "websocket.disconnect":
                break

            raw_bytes = None
            client_sr = None

            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                txt = message["text"].strip()
                if txt == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    continue
                elif txt.startswith("{"):
                    try:
                        parsed = json.loads(txt)
                        if parsed.get("type") == "reset":
                            buf.buffer = buf.buffer[:0]
                            buf.prediction_history.clear()
                            buf.rolling_probability = None
                            await websocket.send_text(json.dumps({"type": "reset_ack"}))
                            continue
                        if "audio_base64" in parsed:
                            raw_bytes = base64.b64decode(parsed["audio_base64"])
                            client_sr = parsed.get("sample_rate")
                        elif "samples" in parsed:
                            raw_bytes = None
                            samples_list = parsed["samples"]
                            client_sr = parsed.get("sample_rate", 16000)
                            audio_np, _ = preprocess_audio(samples_list, sample_rate=client_sr, target_sample_rate=16000)
                            buf.add_samples(audio_np)
                    except Exception as e:
                        logger.warning(f"Failed to parse text audio message: {e}")
                        continue

            if raw_bytes is not None and len(raw_bytes) > 0:
                try:
                    audio_np, _ = preprocess_audio(raw_bytes, sample_rate=client_sr, target_sample_rate=16000)
                    buf.add_samples(audio_np)
                except Exception as e:
                    logger.debug(f"Chunk decode note: {e}")
                    continue

            # Process all complete 3.0s windows
            windows = buf.extract_ready_windows()

            for win_id, win_audio, telemetry in windows:
                synth_res = synthetic_detector.detect(win_audio, sr=16000)

                if synth_res["status"] == "insufficient_speech":
                    pred_entry = buf.record_prediction(
                        window_id=win_id,
                        synthetic_probability=None,
                        speaker_similarity=None,
                        speech_detected=False,
                        vocoder_type="Silence / Ambient",
                        confidence=None,
                        latency_ms=synth_res["processing_time_ms"],
                        reasons=synth_res.get("reason_tags", [])
                    )

                    resp = {
                        "type": "inference_result",
                        "call_id": call_id,
                        "window_id": win_id,
                        "status": "insufficient_speech",
                        "speech_detected": False,
                        "prediction": None,
                        "instant_probability": None,
                        "rolling_probability": buf.rolling_probability,
                        "model_confidence": None,
                        "vocoder_type": "Silence / Ambient",
                        "latency_ms": synth_res["processing_time_ms"],
                        "telemetry": telemetry,
                        "timeline": buf.get_timeline()
                    }
                    await websocket.send_text(json.dumps(resp))
                    await manager.broadcast_telemetry(call_id, resp)
                    continue

                # Speech detected: run verification, replay, behavioral, context, and risk
                chunk_embedding = synth_res.get("embedding", [])
                speaker_similarity = 0.50
                speaker_mismatch = 0.50
                if vp_embedding and chunk_embedding:
                    spk_res = speaker_verifier.verify_similarity(vp_embedding, chunk_embedding)
                    speaker_similarity = spk_res["match_score"]
                    speaker_mismatch = spk_res["mismatch_score"]

                liveness_res = liveness_detector.detect_replay(win_audio, sr=16000)
                replay_score = liveness_res["replay_probability"]

                behavior_res = behavior_analyzer.analyze(synth_res.get("acoustic_metrics", {}))
                behavior_score = behavior_res["behavior_anomaly_score"]

                context_res = context_engine.evaluate(
                    caller_phone=call.caller_id if call else None,
                    registered_phone=registered_phone
                )
                context_score = context_res["context_risk_score"]

                all_tags = []
                all_tags.extend(synth_res.get("reason_tags", []))
                all_tags.extend(liveness_res.get("flags", []))
                all_tags.extend(behavior_res.get("behavioral_flags", []))

                risk_res = risk_engine.calculate_risk(
                    synthetic_prob=synth_res["synthetic_probability"],
                    speaker_mismatch=speaker_mismatch,
                    behavior_score=behavior_score,
                    context_score=context_score,
                    replay_prob=replay_score,
                    collected_tags=all_tags
                )

                pred_entry = buf.record_prediction(
                    window_id=win_id,
                    synthetic_probability=synth_res["synthetic_probability"],
                    speaker_similarity=speaker_similarity,
                    speech_detected=True,
                    vocoder_type=synth_res.get("vocoder_type", "Natural Human"),
                    confidence=synth_res.get("model_confidence"),
                    latency_ms=synth_res["processing_time_ms"],
                    reasons=risk_res["reason_tags"]
                )

                resp = {
                    "type": "inference_result",
                    "call_id": call_id,
                    "window_id": win_id,
                    "status": "success",
                    "speech_detected": True,
                    "prediction": synth_res["prediction"],
                    "instant_probability": synth_res["synthetic_probability"],
                    "rolling_probability": buf.rolling_probability,
                    "model_confidence": synth_res.get("model_confidence"),
                    "speaker_similarity": speaker_similarity,
                    "replay_probability": replay_score,
                    "behavior_anomaly_score": behavior_score,
                    "context_risk_score": context_score,
                    "risk_score": risk_res["risk_score"],
                    "risk_level": risk_res["risk_level"],
                    "recommended_action": risk_res["recommended_action"],
                    "action_description": risk_res["action_description"],
                    "reasons": risk_res["reason_tags"],
                    "vocoder_type": synth_res["vocoder_type"],
                    "latency_ms": synth_res["processing_time_ms"],
                    "telemetry": telemetry,
                    "timeline": buf.get_timeline()
                }

                await websocket.send_text(json.dumps(resp))
                await manager.broadcast_telemetry(call_id, resp)

    except (WebSocketDisconnect, RuntimeError):
        logger.info(f"Live microphone WebSocket disconnected for {call_id}")
    finally:
        db.close()
