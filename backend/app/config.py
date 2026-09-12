"""
VoiceGuard Configuration and Environment Management
Problem Statement SIH26104: AI-Powered Real-Time Voice Cloning Detection and Prevention
"""

import os

class Settings:
    PROJECT_NAME: str = "VoiceGuard Gateway"
    VERSION: str = "2.1.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./voiceguard.db")
    
    # Privacy by Design
    RAW_AUDIO_RETENTION: bool = os.getenv("RAW_AUDIO_RETENTION", "false").lower() == "true"
    FEATURE_LOGGING: bool = True
    
    # ML Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "D:/ML model VG" if os.path.isdir("D:/ML model VG") else "facebook/wav2vec2-base")
    CHECKPOINT_PATH: str = os.getenv("CHECKPOINT_PATH", "D:/voiceguard/checkpoints/best_voiceguard_wav2vec2.pt")
    
    # Dynamic Risk Engine Default Weights (Sum = 1.0)
    WEIGHT_SYNTHETIC: float = 0.35
    WEIGHT_SPEAKER_MISMATCH: float = 0.25
    WEIGHT_BEHAVIOR: float = 0.15
    WEIGHT_CONTEXT: float = 0.15
    WEIGHT_REPLAY: float = 0.10
    
    # Risk Thresholds
    RISK_THRESHOLD_LOW: float = 30.0
    RISK_THRESHOLD_HIGH: float = 70.0
    KILL_SWITCH_THRESHOLD: float = 75.0

settings = Settings()