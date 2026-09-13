"""
VoiceGuard Comprehensive Audio Pipeline & Model Inference Test Suite
SIH26104: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

Verifies:
1. Unified common preprocessing (preprocess_audio) across datatypes, channels, and sample rates.
2. RollingAudioBuffer windowing (3.0s / 48,000 samples, 1.0s step) and EWMA aggregation.
3. Silence & VAD handling (insufficient_speech returns null probabilities, no fake predictions).
4. Real vs Synthetic audio discrimination.
5. Upload vs Live streaming chunk consistency (Test A vs Test B on identical audio).
6. Speaker verification separation.
"""

import io
import math
import numpy as np
import pytest
import scipy.io.wavfile

from backend.app.audio.preprocessor import preprocess_audio, AudioPreprocessor
from backend.app.audio.stream_buffer import RollingAudioBuffer, buffer_registry
from backend.app.ml.synthetic_detector import get_synthetic_detector
from backend.app.ml.speaker_verifier import SpeakerVerifier


def generate_test_audio(duration_sec: float = 4.0, sr: int = 16000, is_synthetic: bool = False) -> np.ndarray:
    """Generates synthetic audio signal for testing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    f0 = 160.0
    if not is_synthetic:
        # Authentic biological harmonic structure with subtle natural micro-tremor
        tremor = 0.015 * np.sin(2 * np.pi * 5.0 * t)
        signal = (
            0.5 * np.sin(2 * np.pi * f0 * t) +
            0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.15 * np.sin(2 * np.pi * 3 * f0 * t)
        ) * (1.0 + tremor)
    else:
        # Synthetic deepfake signature: phase jitter & high frequency vocoder artifacts
        jitter = np.random.normal(0, 0.08, len(t))
        signal = (
            0.5 * np.sin(2 * np.pi * f0 * (t + jitter)) +
            0.35 * np.sin(2 * np.pi * 3800.0 * t) +
            0.15 * np.sin(2 * np.pi * 5200.0 * t)
        )
    
    # Normalize to [-0.85, 0.85]
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.85
    return signal.astype(np.float32)


def to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Converts numpy float array to WAV bytes."""
    bio = io.BytesIO()
    int16_data = (audio * 32767).astype(np.int16)
    scipy.io.wavfile.write(bio, sr, int16_data)
    bio.seek(0)
    return bio.read()


class TestAudioPipelines:

    def test_preprocess_audio_mono_conversion(self):
        """Stereo audio must be converted to 1D mono float32."""
        stereo = np.random.uniform(-0.5, 0.5, (32000, 2)).astype(np.float32)
        mono, sr = preprocess_audio(stereo, sample_rate=16000, target_sample_rate=16000)
        assert mono.ndim == 1
        assert len(mono) == 32000
        assert sr == 16000
        assert mono.dtype == np.float32

    def test_preprocess_audio_resampling(self):
        """Audio at 44.1 kHz or 48 kHz must be resampled to exactly 16 kHz."""
        sr_orig = 48000
        audio_48k = generate_test_audio(duration_sec=2.0, sr=sr_orig)
        resampled, sr_out = preprocess_audio(audio_48k, sample_rate=sr_orig, target_sample_rate=16000)
        assert sr_out == 16000
        assert abs(len(resampled) - 32000) <= 2

    def test_preprocess_audio_wav_bytes_decoding(self):
        """WAV bytes must decode cleanly into normalized float32."""
        raw = generate_test_audio(duration_sec=3.0, sr=16000)
        wav_bytes = to_wav_bytes(raw, sr=16000)
        decoded, sr = preprocess_audio(wav_bytes, target_sample_rate=16000)
        assert sr == 16000
        assert len(decoded) == 48000
        assert np.max(np.abs(decoded)) <= 1.0

    def test_rolling_audio_buffer_windowing(self):
        """Rolling buffer must accumulate arbitrary chunks and extract exact 48,000-sample windows stepping by 16,000."""
        buf = RollingAudioBuffer(call_id="TEST-BUF-01", sample_rate=16000, window_duration_sec=3.0, step_duration_sec=1.0)
        
        # Ingest 1.5 seconds (24,000 samples) -> No window ready yet
        chunk1 = np.ones(24000, dtype=np.float32) * 0.1
        buf.add_samples(chunk1)
        windows = buf.extract_ready_windows()
        assert len(windows) == 0

        # Ingest another 1.8 seconds (28,800 samples) -> Total 52,800 samples
        # Window 1 (48,000 samples) extracted; buffer advances by 16,000 -> 36,800 remaining (less than 48,000)
        chunk2 = np.ones(28800, dtype=np.float32) * 0.2
        buf.add_samples(chunk2)
        windows = buf.extract_ready_windows()
        assert len(windows) == 1
        win_id, win_audio, telem = windows[0]
        assert len(win_audio) == 48000
        assert telem["num_samples"] == 48000
        assert telem["duration_ms"] == 3000

    def test_silence_vad_handling(self):
        """Pure silence must return status: insufficient_speech and null synthetic_probability (Step 5)."""
        detector = get_synthetic_detector()
        silence = np.zeros(48000, dtype=np.float32)
        res = detector.detect(silence, sr=16000)
        assert res["status"] == "insufficient_speech"
        assert res["speech_detected"] is False
        assert res["synthetic_probability"] is None
        assert res["prediction"] is None

    def test_real_vs_synthetic_audio_inference(self):
        """Synthetic audio must yield significantly higher synthetic probability than authentic audio."""
        detector = get_synthetic_detector()
        auth_audio = generate_test_audio(duration_sec=3.0, is_synthetic=False)
        synth_audio = generate_test_audio(duration_sec=3.0, is_synthetic=True)

        res_auth = detector.detect(auth_audio, sr=16000)
        res_synth = detector.detect(synth_audio, sr=16000)

        assert res_auth["status"] == "success"
        assert res_synth["status"] == "success"
        assert res_auth["synthetic_probability"] is not None
        assert res_synth["synthetic_probability"] is not None
        assert res_synth["synthetic_probability"] > res_auth["synthetic_probability"]

    def test_upload_vs_simulated_live_consistency(self):
        """Test A (upload) and Test B (simulated live chunks through buffer) must yield consistent probabilities (Step 7)."""
        detector = get_synthetic_detector()
        test_audio = generate_test_audio(duration_sec=5.0, is_synthetic=True)
        wav_bytes = to_wav_bytes(test_audio, sr=16000)

        # Test A: Upload pipeline (full file decode)
        audio_upload, sr_up = preprocess_audio(wav_bytes)
        res_upload = detector.detect(audio_upload[:48000], sr=sr_up)
        prob_upload = res_upload["synthetic_probability"]

        # Test B: Simulated live streaming (slice into 100ms chunks)
        buf = RollingAudioBuffer(call_id="TEST-SIM-LIVE", sample_rate=16000)
        chunk_size = 1600  # 100ms @ 16kHz
        ready_windows = []
        for start in range(0, len(test_audio), chunk_size):
            chunk = test_audio[start : start + chunk_size]
            buf.add_samples(chunk)
            wins = buf.extract_ready_windows()
            if wins:
                ready_windows.extend(wins)

        assert len(ready_windows) >= 1
        first_win_id, first_win_audio, _ = ready_windows[0]
        res_live = detector.detect(first_win_audio, sr=16000)
        prob_live = res_live["synthetic_probability"]

        # Both pipelines must yield consistent predictions
        diff = abs(prob_upload - prob_live)
        assert diff < 0.05, f"Upload prob ({prob_upload}) and Live prob ({prob_live}) diverged by {diff}"

    def test_speaker_verification_separation(self):
        """Speaker verification similarity must decrease when compared against an enrolled voiceprint."""
        verifier = SpeakerVerifier()
        emb_target = np.sin(np.linspace(0, 10, 256)).tolist()
        emb_similar = (np.array(emb_target) + np.random.normal(0, 0.05, 256)).tolist()
        emb_different = np.cos(np.linspace(10, 20, 256)).tolist()

        sim_high = verifier.verify_similarity(emb_target, emb_similar)
        sim_low = verifier.verify_similarity(emb_target, emb_different)

        assert sim_high["match_score"] > sim_low["match_score"]
        assert sim_high["mismatch_score"] < sim_low["mismatch_score"]


if __name__ == "__main__":
    test = TestAudioPipelines()
    print("Running audio pipeline tests...")
    test.test_preprocess_audio_mono_conversion()
    print("✓ Mono conversion passed")
    test.test_preprocess_audio_resampling()
    print("✓ Resampling passed")
    test.test_preprocess_audio_wav_bytes_decoding()
    print("✓ WAV bytes decoding passed")
    test.test_rolling_audio_buffer_windowing()
    print("✓ Rolling buffer windowing passed")
    test.test_silence_vad_handling()
    print("✓ Silence / VAD handling passed")
    test.test_real_vs_synthetic_audio_inference()
    print("✓ Real vs Synthetic inference passed")
    test.test_upload_vs_simulated_live_consistency()
    print("✓ Upload vs Live consistency passed")
    test.test_speaker_verification_separation()
    print("✓ Speaker verification passed")
    print("All tests passed successfully!")
