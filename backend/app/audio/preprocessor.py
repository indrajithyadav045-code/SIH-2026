"""
VoiceGuard Audio Preprocessor & Acoustic Feature Extractor
Handles 16kHz resampling, mono conversion, VAD (Voice Activity Detection),
fixed/streaming chunking, and physical acoustic feature extraction.
"""

import io
import math
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import scipy.signal
import scipy.io.wavfile

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

logger = logging.getLogger("VoiceGuard.AudioPreprocessor")

class AudioPreprocessor:
    """
    Production-grade audio processing pipeline conforming to SIH26104 standards.
    Processes telecom and VoIP streams into normalized 16kHz audio frames.
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sr = target_sample_rate

    def load_audio(self, source: Union[bytes, str, io.BytesIO, np.ndarray], original_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Loads audio from raw bytes, file path, BytesIO, or numpy array.
        Ensures output is 1D float32 normalized to [-1.0, 1.0] at 16kHz.
        """
        if isinstance(source, np.ndarray):
            audio = source.astype(np.float32)
            sr = original_sr or self.target_sr
        elif isinstance(source, (str, bytes, io.BytesIO)):
            if isinstance(source, bytes):
                source = io.BytesIO(source)
            if HAS_SOUNDFILE:
                try:
                    audio, sr = sf.read(source, dtype='float32')
                except Exception as e:
                    logger.warning(f"Soundfile failed: {e}. Falling back to scipy.")
                    if isinstance(source, io.BytesIO):
                        source.seek(0)
                    sr, audio = scipy.io.wavfile.read(source)
            else:
                sr, audio = scipy.io.wavfile.read(source)
        else:
            raise ValueError(f"Unsupported audio source type: {type(source)}")

        # Convert int types to float32 normalized [-1.0, 1.0]
        if np.issubdtype(audio.dtype, np.integer):
            max_val = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / max_val
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Convert multi-channel (stereo) to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample to target sample rate (16kHz)
        if sr != self.target_sr:
            num_samples = int(round(len(audio) * float(self.target_sr) / sr))
            audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
            sr = self.target_sr

        # Normalize and clip
        max_abs = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
        if max_abs > 1.0:
            audio = np.clip(audio, -1.0, 1.0)

        return audio, sr

    def detect_vad(self, audio: np.ndarray, frame_len_ms: int = 30, energy_thresh: float = 0.005) -> Dict[str, Any]:
        """
        Energy and Zero-Crossing based Voice Activity Detection (VAD).
        Filters silence/noise bursts to ensure valid speech before deep learning inference.
        """
        if len(audio) == 0:
            return {"has_speech": False, "speech_ratio": 0.0, "active_frames": 0, "total_frames": 0}

        frame_size = int(self.target_sr * (frame_len_ms / 1000.0))
        if frame_size <= 0:
            frame_size = 480

        num_frames = len(audio) // frame_size
        if num_frames == 0:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            return {"has_speech": rms > energy_thresh, "speech_ratio": 1.0 if rms > energy_thresh else 0.0, "active_frames": 1 if rms > energy_thresh else 0, "total_frames": 1}

        active_count = 0
        for i in range(num_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms > energy_thresh:
                active_count += 1

        ratio = active_count / float(num_frames)
        return {
            "has_speech": ratio > 0.15,
            "speech_ratio": round(ratio, 4),
            "active_frames": active_count,
            "total_frames": num_frames
        }

    def chunk_audio(self, audio: np.ndarray, chunk_duration_sec: float = 2.0, overlap_sec: float = 0.5) -> List[np.ndarray]:
        """
        Splits a continuous stream into fixed-duration chunks for sequential window scoring.
        """
        chunk_samples = int(self.target_sr * chunk_duration_sec)
        hop_samples = int(self.target_sr * (chunk_duration_sec - overlap_sec))
        if hop_samples <= 0:
            hop_samples = chunk_samples

        if len(audio) <= chunk_samples:
            # Pad with silence if slightly shorter than 1 chunk
            if len(audio) < chunk_samples:
                padded = np.pad(audio, (0, chunk_samples - len(audio)), mode='constant')
                return [padded]
            return [audio]

        chunks = []
        for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
            chunks.append(audio[start : start + chunk_samples])

        # If remainder exists and is significant (>0.5s), pad and include
        rem = len(audio) % hop_samples
        if rem > int(self.target_sr * 0.5) and (len(audio) - rem) < len(audio):
            last_chunk = audio[-chunk_samples:] if len(audio) >= chunk_samples else np.pad(audio, (0, chunk_samples - len(audio)))
            chunks.append(last_chunk)

        return chunks

    def extract_acoustic_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extracts physical acoustic signal features:
        - RMS Energy
        - Zero-Crossing Rate (ZCR)
        - Spectral Centroid & Rolloff
        - Spectral Flatness
        - Pitch & Pitch Jitter estimation
        """
        if len(audio) < 160:
            return {
                "rms_energy": 0.0,
                "zero_crossing_rate": 0.0,
                "spectral_centroid_hz": 0.0,
                "spectral_rolloff_hz": 0.0,
                "spectral_flatness": 0.0,
                "pitch_hz": 0.0,
                "jitter_percent": 0.0
            }

        # RMS
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Zero Crossing Rate
        signs = np.sign(audio)
        signs[signs == 0] = 1
        zcr = float(np.mean(np.abs(np.diff(signs)) > 0))

        # FFT Spectrum
        window = np.hanning(len(audio))
        fft_spec = np.abs(np.fft.rfft(audio * window))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.target_sr)

        # Spectral Centroid
        spec_sum = np.sum(fft_spec)
        centroid = float(np.sum(freqs * fft_spec) / (spec_sum + 1e-12))

        # Spectral Rolloff (85% energy)
        cumulative = np.cumsum(fft_spec)
        rolloff_idx = np.searchsorted(cumulative, 0.85 * spec_sum)
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # Spectral Flatness (Geometric Mean / Arithmetic Mean)
        power_spec = fft_spec ** 2 + 1e-12
        geometric_mean = np.exp(np.mean(np.log(power_spec)))
        arithmetic_mean = np.mean(power_spec)
        flatness = float(geometric_mean / (arithmetic_mean + 1e-12))

        # Pitch & Jitter via Autocorrelation
        pitch, jitter = self._estimate_pitch_and_jitter(audio)

        return {
            "rms_energy": round(rms, 4),
            "zero_crossing_rate": round(zcr, 4),
            "spectral_centroid_hz": round(centroid, 1),
            "spectral_rolloff_hz": round(rolloff, 1),
            "spectral_flatness": round(flatness, 4),
            "pitch_hz": round(pitch, 1),
            "jitter_percent": round(jitter, 3)
        }

    def _estimate_pitch_and_jitter(self, audio: np.ndarray) -> Tuple[float, float]:
        """Autocorrelation-based fundamental frequency (F0) and period perturbation (jitter)."""
        min_lag = int(self.target_sr / 400.0)  # max pitch 400Hz
        max_lag = int(self.target_sr / 60.0)   # min pitch 60Hz

        if len(audio) < max_lag * 2:
            return 0.0, 0.0

        # Segment into three sub-frames to measure pitch stability
        sub_len = len(audio) // 3
        pitches = []
        for s in range(3):
            sub = audio[s * sub_len : (s + 1) * sub_len]
            corr = np.correlate(sub, sub, mode='full')
            corr = corr[len(corr) // 2 :]
            if len(corr) > max_lag:
                lag_window = corr[min_lag:max_lag]
                if len(lag_window) > 0 and np.max(lag_window) > 0.3 * corr[0]:
                    best_lag = min_lag + np.argmax(lag_window)
                    pitches.append(self.target_sr / float(best_lag))

        if not pitches:
            return 0.0, 0.0

        mean_pitch = float(np.mean(pitches))
        jitter = float(np.std(pitches) / (mean_pitch + 1e-6) * 100.0)
        return mean_pitch, jitter