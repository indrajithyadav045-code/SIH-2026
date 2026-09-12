"""
VoiceGuard Call Lifecycle & Real-Time Audio Chunk Analysis Endpoints
"""

import uuid
import datetime
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.schema_models import Call, User, VoiceProfile, RiskEvent, Alert
from backend.app.schemas.api_schemas import (
    CallStartRequest, CallResponse, RiskEventResponse, CallSummaryResponse
)
from backend.app.audio.preprocessor import AudioPreprocessor
from backend.app.ml.synthetic_detector import get_synthetic_detector
from backend.app.ml.speaker_verifier import SpeakerVerifier
from backend.app.ml.liveness_detector import LivenessDetector
from backend.app.risk.behavior_analyzer import BehaviorAnalyzer
from backend.app.risk.context_engine import ContextEngine
from backend.app.risk.risk_engine import RiskEngine

logger = logging.getLogger("VoiceGuard.API.Calls")
router = APIRouter(prefix="/calls", tags=["Calls"])

preprocessor = AudioPreprocessor(target_sample_rate=16000)
synthetic_detector = get_synthetic_detector()
speaker_verifier = SpeakerVerifier()
liveness_detector = LivenessDetector()
behavior_analyzer = BehaviorAnalyzer()
context_engine = ContextEngine()
risk_engine = RiskEngine()

@router.post("/start", response_model=CallResponse, status_code=status.HTTP_201_CREATED)
def start_call(payload: CallStartRequest, db: Session = Depends(get_db)):
    """Initiates a new voice session for live monitoring."""
    call_id = f"CALL-{uuid.uuid4().hex[:8].upper()}"
    new_call = Call(
        id=call_id,
        caller_id=payload.caller_id,
        claimed_identity=payload.claimed_identity,
        channel=payload.channel,
        status="IN_PROGRESS",
        final_risk_score=5.0,
        final_risk_level="LOW"
    )
    db.add(new_call)
    db.commit()
    db.refresh(new_call)
    return new_call

@router.get("", response_model=List[CallResponse])
def list_calls(limit: int = 25, db: Session = Depends(get_db)):
    """Lists recent calls with current risk ratings."""
    return db.query(Call).order_by(Call.start_time.desc()).limit(limit).all()

@router.get("/{call_id}", response_model=CallSummaryResponse)
def get_call_summary(call_id: str, db: Session = Depends(get_db)):
    """Retrieves call details, risk timeline events, and current defensive posture."""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call session not found")
    
    events = db.query(RiskEvent).filter(RiskEvent.call_id == call_id).order_by(RiskEvent.timestamp.asc()).all()

    return {
        "id": call.id,
        "caller_id": call.caller_id,
        "claimed_identity": call.claimed_identity,
        "channel": call.channel,
        "status": call.status,
        "final_risk_score": call.final_risk_score,
        "final_risk_level": call.final_risk_level,
        "start_time": call.start_time,
        "end_time": call.end_time,
        "risk_events_count": len(events),
        "recent_events": events[-10:]
    }

@router.post("/{call_id}/analyze_chunk", response_model=RiskEventResponse)
async def analyze_audio_chunk(
    call_id: str,
    audio_file: UploadFile = File(...),
    transcript: Optional[str] = Form(None),
    transaction_amount: Optional[float] = Form(None),
    is_new_beneficiary: bool = Form(False),
    db: Session = Depends(get_db)
):
    """
    Core Pipeline Entrypoint:
    Executes real-time 10-stage evaluation on incoming audio chunk.
    """
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call session not found")

    audio_bytes = await audio_file.read()
    try:
        audio_np, sr = preprocessor.load_audio(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio file: {e}")

    # 1. Synthetic Detection
    synth_res = synthetic_detector.detect(audio_np, sr=sr)
    chunk_embedding = synth_res.get("embedding", [])

    # 2. Speaker Verification
    speaker_mismatch = 0.50
    speaker_similarity = 0.50
    registered_phone = None

    if call.claimed_identity:
        # Search by ID or Name
        user = db.query(User).filter((User.id == call.claimed_identity) | (User.name == call.claimed_identity)).first()
        if user:
            registered_phone = user.registered_phone
            vp = db.query(VoiceProfile).filter(VoiceProfile.user_id == user.id).first()
            if vp and vp.embedding:
                spk_res = speaker_verifier.verify_similarity(vp.embedding, chunk_embedding)
                speaker_similarity = spk_res["match_score"]
                speaker_mismatch = spk_res["mismatch_score"]

    # 3. Liveness & Replay
    liveness_res = liveness_detector.detect_replay(audio_np, sr=sr)
    replay_score = liveness_res["replay_probability"]

    # 4. Behavioral Analysis
    behavior_res = behavior_analyzer.analyze(synth_res.get("acoustic_metrics", {}), transcript=transcript)
    behavior_score = behavior_res["behavior_anomaly_score"]

    # 5. Context Engine
    context_res = context_engine.evaluate(
        caller_phone=call.caller_id,
        registered_phone=registered_phone,
        transaction_amount=transaction_amount,
        is_new_beneficiary=is_new_beneficiary
    )
    context_score = context_res["context_risk_score"]

    # Aggregate Reason Tags
    all_tags = []
    all_tags.extend(synth_res.get("reason_tags", []))
    all_tags.extend(liveness_res.get("flags", []))
    all_tags.extend(behavior_res.get("behavioral_flags", []))
    all_tags.extend(context_res.get("context_tags", []))

    # 6. Risk Engine
    risk_res = risk_engine.calculate_risk(
        synthetic_prob=synth_res["synthetic_probability"],
        speaker_mismatch=speaker_mismatch,
        behavior_score=behavior_score,
        context_score=context_score,
        replay_prob=replay_score,
        collected_tags=all_tags
    )

    # Update Call
    call.final_risk_score = risk_res["risk_score"]
    call.final_risk_level = risk_res["risk_level"]
    if risk_res["kill_switch_triggered"]:
        call.status = "FLAGGED_THREAT"

    # Count prior chunks
    chunk_count = db.query(RiskEvent).filter(RiskEvent.call_id == call_id).count()

    # 7. Persist Risk Event
    new_event = RiskEvent(
        call_id=call_id,
        chunk_index=chunk_count + 1,
        timestamp=datetime.datetime.utcnow(),
        synthetic_score=synth_res["synthetic_probability"],
        speaker_similarity=speaker_similarity,
        speaker_mismatch=speaker_mismatch,
        behavior_score=behavior_score,
        context_score=context_score,
        replay_score=replay_score,
        risk_score=risk_res["risk_score"],
        risk_level=risk_res["risk_level"],
        reasons=risk_res["reason_tags"]
    )
    db.add(new_event)

    # If High Risk, generate alert
    if risk_res["risk_level"] == "HIGH" or risk_res["kill_switch_triggered"]:
        alert = Alert(
            id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
            call_id=call_id,
            recipient_role="FRAUD_OPS",
            severity="CRITICAL" if risk_res["kill_switch_triggered"] else "WARNING",
            message=f"Impersonation Threat: Risk {risk_res['risk_score']}% on Call {call_id}. {risk_res['action_description']}",
            status="ACTIVE"
        )
        db.add(alert)

    db.commit()
    db.refresh(new_event)
    return new_event

@router.post("/{call_id}/end", response_model=CallResponse)
def end_call(call_id: str, db: Session = Depends(get_db)):
    """Ends the call session."""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call session not found")
    call.status = "TERMINATED"
    call.end_time = datetime.datetime.utcnow()
    db.commit()
    db.refresh(call)
    return call