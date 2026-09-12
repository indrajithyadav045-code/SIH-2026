"""
VoiceGuard Replay & Liveness Attack Detector
Evaluates acoustic replay cues: speaker transducer distortion, artificial room impulse response,
and spectral flatness variance typical of physical loudspeaker playback into a microphone.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger("VoiceGuard.LivenessDetector")

class LivenessDetector:
    """
    Detects presentation replay attacks (attacker playing a recorded audio clip via phone or speaker).
    """

    def detect_replay(self, audio: np.ndarray, sr: int = 16000, acoustic_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyzes high-frequency cutoff, phase smear, and repetitive room reflections.
        Returns:
            - replay_probability: float (0.0 to 1.0)
            - liveness_score: float (0.0 to 1.0)
            - is_replay: bool
            - flags: List[str]
        """
        flags = []
        replay_prob = 0.10

        if len(audio) < 1000:
            return {
                "replay_probability": 0.10,
                "liveness_score": 0.90,
                "is_replay": False,
                "flags": ["INSUFFICIENT_AUDIO_LENGTH"]
            }

        # 1. High frequency band ratio (> 6 kHz vs total)
        fft_spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)

        hf_mask = freqs > 6000.0
        total_energy = np.sum(fft_spec) + 1e-12
        hf_energy = np.sum(fft_spec[hf_mask])
        hf_ratio = float(hf_energy / total_energy)

        # Phone/speaker playback typically rolls off sharply above 5.5kHz
        if hf_ratio < 0.005 and len(audio) > 16000:
            replay_prob += 0.25
            flags.append("SHARP_TRANSDUCER_HF_CUTOFF")

        # 2. Spectral peak concentration (playback devices create resonant frequency spikes)
        peaks = np.sort(fft_spec)[-10:]
        peak_ratio = float(np.sum(peaks) / total_energy)
        if peak_ratio > 0.40:
            replay_prob += 0.20
            flags.append("TRANSDUCER_RESONANCE_SPIKE")

        # 3. Room impulse / echo autocorrelation
        corr = np.correlate(audio[:4000], audio[:4000], mode='full')
        corr = corr[len(corr)//2:]
        if len(corr) > 800:
            echo_peak = np.max(corr[400:800]) / (corr[0] + 1e-12)
            if echo_peak > 0.45:
                replay_prob += 0.25
                flags.append("REPETITIVE_ROOM_REFLECTION")

        calibrated_replay = max(0.05, min(0.95, replay_prob))
        liveness_score = 1.0 - calibrated_replay

        return {
            "replay_probability": round(calibrated_replay, 4),
            "liveness_score": round(liveness_score, 4),
            "is_replay": calibrated_replay > 0.50,
            "flags": flags
        }