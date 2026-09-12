"""
VoiceGuard Identity & Biometric Profile Management
"""

import uuid
import datetime
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.schema_models import User, VoiceProfile
from backend.app.schemas.api_schemas import UserProfileResponse
from backend.app.audio.preprocessor import AudioPreprocessor
from backend.app.ml.synthetic_detector import get_synthetic_detector

logger = logging.getLogger("VoiceGuard.API.Profiles")
router = APIRouter(prefix="/profiles", tags=["Profiles"])

preprocessor = AudioPreprocessor(target_sample_rate=16000)
synthetic_detector = get_synthetic_detector()

@router.get("", response_model=List[UserProfileResponse])
def list_profiles(db: Session = Depends(get_db)):
    """Lists all registered institutional users and biometric enrollment status."""
    users = db.query(User).all()
    results = []
    for u in users:
        vp = db.query(VoiceProfile).filter(VoiceProfile.user_id == u.id).first()
        results.append({
            "id": u.id,
            "name": u.name,
            "role": u.role,
            "organization": u.organization,
            "registered_phone": u.registered_phone,
            "is_enrolled": vp is not None,
            "created_at": u.created_at
        })
    return results

@router.post("/enroll", status_code=status.HTTP_201_CREATED)
async def enroll_voice_profile(
    user_id: str = Form(...),
    language: str = Form("en"),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Enrolls a clean reference voiceprint (256-dim embedding) for an authorized user."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    audio_bytes = await audio_file.read()
    try:
        audio_np, sr = preprocessor.load_audio(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read enrollment audio: {e}")

    vad = preprocessor.detect_vad(audio_np)
    if not vad["has_speech"]:
        raise HTTPException(status_code=400, detail="No speech detected in enrollment audio.")

    res = synthetic_detector.detect(audio_np, sr=sr)
    emb = res.get("embedding")

    vp = db.query(VoiceProfile).filter(VoiceProfile.user_id == user_id).first()
    if not vp:
        vp = VoiceProfile(
            id=f"VP-{uuid.uuid4().hex[:8].upper()}",
            user_id=user_id,
            embedding=emb,
            language=language,
            sample_count=1
        )
        db.add(vp)
    else:
        vp.embedding = emb
        vp.sample_count += 1
        vp.enrolled_at = datetime.datetime.utcnow()

    db.commit()
    db.refresh(vp)
    return {
        "status": "SUCCESS",
        "message": f"Biometric voice profile enrolled for {user.name}",
        "profile_id": vp.id,
        "sample_count": vp.sample_count
    }