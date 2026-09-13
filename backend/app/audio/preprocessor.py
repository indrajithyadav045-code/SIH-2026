"""
VoiceGuard Real-Time Audio Preprocessor & Feature Extraction Pipeline
SIH26104: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

Converts all incoming audio (uploaded files, WebM streams, raw PCM, or microphone buffers)
into normalized 16 kHz Mono Float32 waveforms [-1.0, 1.0].
"""

import io
import math
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import scipy.signal
import scipy.io.wavfile

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

logger = logging.getLogger("VoiceGuard.AudioPreprocessor")


def preprocess_audio(
    source: Union[str, bytes, io.BytesIO, np.ndarray, list],
    sample_rate: Optional[int] = None,
    target_sample_rate: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Unified Common Preprocessing Pipeline for BOTH Uploaded Audio and Live Microphone Streams.
    
    Guarantees Output Format:
    - FORMAT: 1D PCM float32 numpy array
    - CHANNELS: Mono (1)
    - SAMPLE RATE: target_sample_rate (default 16,000 Hz)
    - DTYPE: np.float32
    - RANGE: [-1.0, 1.0]
    """
    sr = sample_rate

    # 1. Decode raw input into numpy array
    if isinstance(source, list):
        audio = np.array(source, dtype=np.float32)
        sr = sr or target_sample_rate
    elif isinstance(source, np.ndarray):
        audio = source.copy()
        sr = sr or target_sample_rate
    elif isinstance(source, (str, bytes, io.BytesIO)):
        if isinstance(source, bytes):
            bio = io.BytesIO(source)
        elif isinstance(source, str):
            bio = source
        else:
            bio = source

        decoded = False
        # Try soundfile first (handles WAV, FLAC, OGG, MP3)
        if HAS_SOUNDFILE:
            try:
                if isinstance(bio, io.BytesIO):
                    bio.seek(0)
                audio, sr = sf.read(bio, dtype='float32')
                decoded = True
            except Exception as e_sf:
                logger.debug(f"Soundfile decode failed: {e_sf}. Trying alternatives...")

        # Fallback to torchaudio (handles WebM, Opus, MP3, etc.)
        if not decoded and HAS_TORCHAUDIO:
            try:
                if isinstance(bio, io.BytesIO):
                    bio.seek(0)
                waveform, sr_torch = torchaudio.load(bio)
                audio = waveform.squeeze().cpu().numpy()
                sr = sr_torch
                decoded = True
            except Exception as e_ta:
                logger.debug(f"Torchaudio decode failed: {e_ta}. Trying scipy...")

        # Fallback to scipy.io.wavfile
        if not decoded:
            try:
                if isinstance(bio, io.BytesIO):
                    bio.seek(0)
                sr_scipy, raw_audio = scipy.io.wavfile.read(bio)
                audio = raw_audio
                sr = sr_scipy
                decoded = True
            except Exception as e_scipy:
                # Raw PCM buffer fallback (e.g. 16-bit PCM or 32-bit float bytes)
                if isinstance(source, bytes) and len(source) >= 2:
                    try:
                        # Try float32
                        audio = np.frombuffer(source, dtype=np.float32).copy()
                        sr = sr or target_sample_rate
                        decoded = True
                    except Exception:
                        try:
                            # Try int16
                            audio = np.frombuffer(source, dtype=np.int16).copy()
                            sr = sr or target_sample_rate
                            decoded = True
                        except Exception:
                            pass

        if not decoded:
            raise ValueError(f"Could not decode audio input of type {type(source)}")
    else:
        raise ValueError(f"Unsupported audio source type: {type(source)}")

    # 2. Convert integer datatypes to float32 normalized in [-1.0, 1.0]
    if np.issubdtype(audio.dtype, np.integer):
        max_val = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_val
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # 3. Convert multi-channel (stereo) to mono
    if audio.ndim > 1:
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
            # (channels, samples) format
            audio = np.mean(audio, axis=0)
        else:
            # (samples, channels) format
            audio = np.mean(audio, axis=1)

    audio = np.ascontiguousarray(audio, dtype=np.float32)

    # 4. Resample to target sample rate (16,000 Hz)
    sr = sr or target_sample_rate
    if sr != target_sample_rate and len(audio) > 0:
        target_num_samples = int(round(len(audio) * float(target_sample_rate) / float(sr)))
        if target_num_samples > 0:
            audio = scipy.signal.resample(audio, target_num_samples).astype(np.float32)
        sr = target_sample_rate

    # 5. Normalize and clip to [-1.0, 1.0]
    if len(audio) > 0:
        max_abs = float(np.max(np.abs(audio)))
        if max_abs > 1.0:
            audio = np.clip(audio, -1.0, 1.0)

    return audio, target_sample_rate


class AudioPreprocessor:
    """
    Audio Preprocessor and DSP Feature Extraction Engine.
    Exposes unified load_audio, VAD, rolling windowing, and acoustic feature analysis.
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sr = target_sample_rate

    def load_audio(
        self,
        source: Union[str, bytes, io.BytesIO, np.ndarray],
        original_sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """Loads and normalizes audio via the common preprocess_audio pipeline."""
        return preprocess_audio(source, sample_rate=original_sr, target_sample_rate=self.target_sr)

    def detect_vad(
        self,
        audio: np.ndarray,
        frame_len_ms: int = 30,
        energy_thresh: float = 0.005
    ) -> Dict[str, Any]:
        """
        Energy and Zero-Crossing based Voice Activity Detection (VAD).
        Filters silence/noise bursts to ensure valid speech before deep learning inference.
        """
        if len(audio) == 0:
            return {
                "has_speech": False,
                "speech_ratio": 0.0,
                "active_frames": 0,
                "total_frames": 0,
                "rms": 0.0
            }

        frame_size = int(self.target_sr * (frame_len_ms / 1000.0))
        if frame_size <= 0:
            frame_size = 480

        num_frames = len(audio) // frame_size
        overall_rms = float(np.sqrt(np.mean(audio ** 2)))

        if num_frames == 0:
            has_speech = overall_rms > energy_thresh
            return {
                "has_speech": has_speech,
                "speech_ratio": 1.0 if has_speech else 0.0,
                "active_frames": 1 if has_speech else 0,
                "total_frames": 1,
                "rms": round(overall_rms, 6)
            }

        active_count = 0
        for i in range(num_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms > energy_thresh:
                active_count += 1

        ratio = active_count / float(num_frames)
        return {
            "has_speech": ratio > 0.15 and overall_rms > energy_thresh,
            "speech_ratio": round(ratio, 4),
            "active_frames": active_count,
            "total_frames": num_frames,
            "rms": round(overall_rms, 6)
        }

    def chunk_audio(
        self,
        audio: np.ndarray,
        chunk_duration_sec: float = 3.0,
        overlap_sec: float = 1.0
    ) -> List[np.ndarray]:
        """
        Splits a continuous stream into fixed-duration chunks (default: 3.0s window, 1.0s overlap).
        """
        chunk_samples = int(self.target_sr * chunk_duration_sec)
        hop_samples = int(self.target_sr * (chunk_duration_sec - overlap_sec))
        if hop_samples <= 0:
            hop_samples = chunk_samples

        if len(audio) <= chunk_samples:
            if len(audio) < chunk_samples:
                padded = np.pad(audio, (0, chunk_samples - len(audio)), mode='constant')
                return [padded]
            return [audio]

        chunks = []
        for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
            chunks.append(audio[start : start + chunk_samples])

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

        # Spectral Flatness
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
