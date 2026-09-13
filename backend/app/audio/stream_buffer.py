"""
VoiceGuard Rolling Audio Buffer & Temporal Windowing Manager
SIH26104: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

Accumulates streaming audio chunks and extracts continuous, overlapping 3.0-second windows
(48,000 samples @ 16kHz, stepping by 1.0s / 16,000 samples).
Maintains rolling prediction history with Exponentially Weighted Moving Average (EWMA).
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger("VoiceGuard.StreamBuffer")


class RollingAudioBuffer:
    """
    Session-specific rolling audio buffer for real-time live microphone evaluation.
    Converts arbitrary live audio chunks into stable 3.0s windows with 1.0s overlap.
    """

    def __init__(
        self,
        call_id: str,
        sample_rate: int = 16000,
        window_duration_sec: float = 3.0,
        step_duration_sec: float = 1.0,
        ewma_alpha: float = 0.40,
        history_max_len: int = 50
    ):
        self.call_id = call_id
        self.sample_rate = sample_rate
        self.window_size = int(round(sample_rate * window_duration_sec))  # 48,000 samples
        self.step_size = int(round(sample_rate * step_duration_sec))      # 16,000 samples
        self.ewma_alpha = ewma_alpha
        self.history_max_len = history_max_len

        # Audio samples buffer (1D float32)
        self.buffer = np.array([], dtype=np.float32)
        self.total_samples_ingested = 0
        self.window_counter = 0

        # Prediction and telemetry history
        self.prediction_history: List[Dict[str, Any]] = []
        self.rolling_probability: Optional[float] = None
        self.rolling_confidence: Optional[float] = None
        self.created_at = time.time()
        self.last_activity = time.time()

    def add_samples(self, new_samples: np.ndarray) -> int:
        """
        Appends new 16 kHz float32 samples to the rolling buffer.
        Returns the new total buffer length.
        """
        if new_samples is None or len(new_samples) == 0:
            return len(self.buffer)

        flat = np.nan_to_num(new_samples.flatten().astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        flat = np.clip(flat, -1.0, 1.0)
        self.buffer = np.concatenate((self.buffer, flat))
        self.total_samples_ingested += len(flat)
        self.last_activity = time.time()

        # Prevent unbounded buffer growth (keep at most 10 seconds in history)
        max_retained = self.window_size + (self.step_size * 5)  # ~8 seconds
        if len(self.buffer) > max_retained:
            # Shift buffer forward
            overflow = len(self.buffer) - max_retained
            self.buffer = self.buffer[overflow:]

        return len(self.buffer)

    def extract_ready_windows(self) -> List[Tuple[int, np.ndarray, Dict[str, Any]]]:
        """
        Extracts all completed 3.0s windows (48,000 samples) ready for inference.
        Advances the buffer read pointer by step_size (16,000 samples) for each window.
        Returns list of tuples: (window_id, window_audio, diagnostic_telemetry).
        """
        ready = []
        while len(self.buffer) >= self.window_size:
            window = self.buffer[:self.window_size].copy()
            self.window_counter += 1
            
            # Diagnostic telemetry for this window
            rms = float(np.sqrt(np.mean(window ** 2)))
            min_val = float(np.min(window))
            max_val = float(np.max(window))
            
            telemetry = {
                "window_id": self.window_counter,
                "call_id": self.call_id,
                "sample_rate": self.sample_rate,
                "channels": 1,
                "duration_ms": int((self.window_size / self.sample_rate) * 1000),
                "num_samples": self.window_size,
                "min_amplitude": round(min_val, 4),
                "max_amplitude": round(max_val, 4),
                "rms": round(rms, 6),
                "speech_detected": rms > 0.005
            }

            ready.append((self.window_counter, window, telemetry))

            # Advance by step size (1.0 second = 16,000 samples)
            self.buffer = self.buffer[self.step_size:]

        return ready

    def record_prediction(
        self,
        window_id: int,
        synthetic_probability: Optional[float],
        speaker_similarity: Optional[float],
        speech_detected: bool,
        vocoder_type: str = "Natural Human",
        confidence: Optional[float] = None,
        latency_ms: float = 0.0,
        reasons: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Records a completed window prediction and computes Exponentially Weighted Moving Average (EWMA).
        """
        if synthetic_probability is not None and speech_detected:
            if self.rolling_probability is None:
                self.rolling_probability = synthetic_probability
            else:
                self.rolling_probability = (
                    self.ewma_alpha * synthetic_probability +
                    (1.0 - self.ewma_alpha) * self.rolling_probability
                )

            if confidence is not None:
                if self.rolling_confidence is None:
                    self.rolling_confidence = confidence
                else:
                    self.rolling_confidence = (
                        self.ewma_alpha * confidence +
                        (1.0 - self.ewma_alpha) * self.rolling_confidence
                    )
        
        elapsed = time.time() - self.created_at
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        time_tag = f"{mins:02d}:{secs:02d}"

        entry = {
            "window_id": window_id,
            "timestamp": time.time(),
            "time_tag": time_tag,
            "speech_detected": speech_detected,
            "instant_probability": round(synthetic_probability, 4) if synthetic_probability is not None else None,
            "rolling_probability": round(self.rolling_probability, 4) if self.rolling_probability is not None else None,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "rolling_confidence": round(self.rolling_confidence, 4) if self.rolling_confidence is not None else None,
            "speaker_similarity": round(speaker_similarity, 4) if speaker_similarity is not None else None,
            "vocoder_type": vocoder_type,
            "latency_ms": round(latency_ms, 2),
            "reasons": reasons or []
        }

        self.prediction_history.append(entry)
        if len(self.prediction_history) > self.history_max_len:
            self.prediction_history.pop(0)

        return entry

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Returns the chronological threat score timeline for UI display."""
        timeline = []
        for p in self.prediction_history:
            if p["rolling_probability"] is not None:
                score_100 = round(p["rolling_probability"] * 100.0, 1)
                timeline.append({
                    "time": p["time_tag"],
                    "score": score_100,
                    "instant": round(p["instant_probability"] * 100.0, 1) if p["instant_probability"] is not None else score_100,
                    "speech": p["speech_detected"]
                })
        return timeline


class SessionBufferRegistry:
    """Registry of active rolling buffers keyed by call_id."""
    def __init__(self):
        self._buffers: Dict[str, RollingAudioBuffer] = {}

    def get_or_create(self, call_id: str) -> RollingAudioBuffer:
        if call_id not in self._buffers:
            self._buffers[call_id] = RollingAudioBuffer(call_id=call_id)
        return self._buffers[call_id]

    def remove(self, call_id: str):
        if call_id in self._buffers:
            del self._buffers[call_id]

    def cleanup_inactive(self, max_idle_sec: float = 300.0):
        now = time.time()
        stale = [cid for cid, buf in self._buffers.items() if (now - buf.last_activity) > max_idle_sec]
        for cid in stale:
            del self._buffers[cid]


buffer_registry = SessionBufferRegistry()
