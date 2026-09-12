"""
VoiceGuard Conversational & Behavioral Anomaly Analyzer
Measures conversational rhythm, pause cadences, speech velocity, and linguistic urgency signals.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger("VoiceGuard.BehaviorAnalyzer")

class BehaviorAnalyzer:
    """
    Evaluates prosodic naturalness, conversational tempo, and social engineering urgency.
    """

    URGENCY_KEYWORDS = [
        "immediate", "urgently", "wire right now", "confidential", "bypassing protocol",
        "don't call me back", "cfo authorized", "off the record", "fast track", "emergency"
    ]

    def analyze(self, acoustic_metrics: Dict[str, Any], transcript: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates behavior anomaly score [0.0, 1.0] and identifies behavioral flags.
        """
        flags = []
        anomaly_score = 0.15

        # 1. Pitch standard deviation & Prosody
        pitch = acoustic_metrics.get("pitch_hz", 0.0)
        jitter = acoustic_metrics.get("jitter_percent", 1.0)
        if pitch > 0:
            if jitter < 0.20:
                anomaly_score += 0.25
                flags.append("MONOTONE_SYNTHETIC_PROSODY")
            elif jitter > 4.5:
                anomaly_score += 0.20
                flags.append("HIGH_EMOTIONAL_STRESS_PERTURBATION")

        # 2. Energy variance (robotic normalization has near-zero RMS dynamics)
        rms = acoustic_metrics.get("rms_energy", 0.0)
        if 0.0 < rms < 0.01:
            anomaly_score += 0.15
            flags.append("UNUSUALLY_LOW_DYNAMIC_RANGE")

        # 3. Transcript social engineering / urgency detection
        if transcript:
            lower_text = transcript.lower()
            matched_keywords = [kw for kw in self.URGENCY_KEYWORDS if kw in lower_text]
            if matched_keywords:
                anomaly_score += 0.35
                flags.append(f"SOCIAL_ENGINEERING_URGENCY: {', '.join(matched_keywords)}")

        final_score = max(0.05, min(0.95, anomaly_score))
        return {
            "behavior_anomaly_score": round(final_score, 4),
            "behavioral_flags": flags
        }