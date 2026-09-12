"""
VoiceGuard Production Inference & Threat Detection Pipeline (SIH26104)
Problem Statement: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

Features:
1. Real-time audio stream / batch processing.
2. Anti-lossy degradation window normalization (3.0s window @ 16 kHz).
3. Temperature-scaled threat probability scoring.
4. Active kill-switch decision engine (blocking threshold, e.g. 75%).
5. Forensic Acoustic Anomaly Explanation:
   - Physical metrics: SNR, Zero-Crossing Rate, Spectral Centroid, RMS, Clipping, Flatness.
   - Anomaly tags: [VOC_PHASE_INCOHERENCE, TELEPHONY_COMPRESSION, CLIPPED_PEAKS, PITCH_FLATNESS, LOW_SNR_DEGRADATION].
   - Synthetic Vocoder Subtype identification: [bona_fide, diffwave, melgan, wavenet, elevenlabs_neural].
   - 256-dim L2-normalized forensic acoustic embedding vector.
"""

import os
import json
import time
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio

from voiceguard_wav2vec2 import VoiceGuardWav2Vec2, extract_forensic_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("VoiceGuard.Inference")


class VoiceGuardInferenceEngine:
    """
    Production-grade VoiceGuard Inference Engine.
    Handles checkpoint loading, real-time waveform preparation, calibrated scoring,
    and forensic anomaly reporting.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        backbone_model_name: str = "D:/ML model VG" if os.path.isdir("D:/ML model VG") else "facebook/wav2vec2-base",
        threshold: float = 0.75,
        temperature: float = 1.0,
        device: Optional[str] = None
    ):
        self.threshold = threshold
        self.temperature = temperature
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Initializing Inference Engine on {self.device} (Threshold: {threshold*100:.1f}%, Temp: {temperature})")

        # 1. Instantiate Model Architecture
        self.model = VoiceGuardWav2Vec2(
            pretrained_model_name=backbone_model_name,
            num_vocoder_classes=5,
            unfreeze_top_n_layers=3,
            temperature=temperature
        ).to(self.device)

        # 2. Load Checkpoint Weights if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading trained weights from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
            if "threshold" in ckpt:
                logger.info(f"Calibrated checkpoint threshold: {ckpt['threshold']:.4f}")
        else:
            logger.info("Running with initialized backbone & projection heads.")

        self.model.eval()

    def prepare_waveform(self, audio_input: Union[str, np.ndarray, torch.Tensor], sr: int = 16000) -> torch.Tensor:
        """Loads and prepares audio into a normalized 3.0s window (48,000 samples @ 16 kHz)."""
        if isinstance(audio_input, str):
            waveform, sample_rate = torchaudio.load(audio_input)
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sample_rate != 16000:
                waveform = F_audio.resample(waveform, sample_rate, 16000)
        elif isinstance(audio_input, np.ndarray):
            waveform = torch.from_numpy(audio_input).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if sr != 16000:
                waveform = F_audio.resample(waveform, sr, 16000)
        elif isinstance(audio_input, torch.Tensor):
            waveform = audio_input.float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if sr != 16000:
                waveform = F_audio.resample(waveform, sr, 16000)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        # Uniform 3.0-second window normalization (48,000 samples)
        target = 48000
        num_samples = waveform.size(-1)
        if num_samples < target:
            waveform = F.pad(waveform, (0, target - num_samples), mode="constant", value=0.0)
        elif num_samples > target:
            # Centered window crop for deterministic evaluation
            start = (num_samples - target) // 2
            waveform = waveform[..., start:start + target]

        return waveform.to(self.device)

    @torch.no_grad()
    def inspect_audio(
        self,
        audio_input: Union[str, np.ndarray, torch.Tensor],
        sr: int = 16000,
        return_embedding: bool = False
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference and forensic anomaly diagnosis.
        Returns:
            Structured dictionary with decision, threat level, vocoder subtype,
            forensic acoustic metrics, and anomaly tags.
        """
        t0 = time.time()
        waveform = self.prepare_waveform(audio_input, sr=sr)
        
        # 1. Model Forward Pass
        outputs = self.model(waveform)
        
        # 2. Calibrated Binary Threat Probability
        scaled_logits = outputs["binary_logits"] / max(self.temperature, 1e-4)
        binary_probs = F.softmax(scaled_logits, dim=-1)[0]
        real_prob = float(binary_probs[0].item())
        fake_prob = float(binary_probs[1].item())
        
        # 3. Vocoder Subtype Prediction
        vocoder_probs = F.softmax(outputs["vocoder_logits"], dim=-1)[0]
        vocoder_idx = int(torch.argmax(vocoder_probs).item())
        vocoder_name = self.model.VOCODER_CLASSES[vocoder_idx]
        vocoder_conf = float(vocoder_probs[vocoder_idx].item())
        
        # 4. Physical Signal Forensic Properties
        forensic_metrics = extract_forensic_metrics(waveform.cpu())
        
        # 5. Threat Level & Kill-Switch Decision
        threat_score = fake_prob
        if threat_score >= 0.85:
            threat_level = "CRITICAL"
        elif threat_score >= self.threshold:
            threat_level = "HIGH"
        elif threat_score >= 0.45:
            threat_level = "SUSPICIOUS"
        else:
            threat_level = "AUTHENTIC"
            
        decision = "KILL_SWITCH_BLOCKED" if threat_score >= self.threshold else "AUTHENTIC_VERIFIED"
        
        # Latency
        inference_latency_ms = round((time.time() - t0) * 1000.0, 2)
        
        result = {
            "decision": decision,
            "threat_score": round(threat_score, 4),
            "threat_level": threat_level,
            "threshold_applied": self.threshold,
            "probabilities": {
                "bona_fide": round(real_prob, 4),
                "spoof_clone": round(fake_prob, 4)
            },
            "vocoder_analysis": {
                "predicted_subtype": vocoder_name,
                "confidence": round(vocoder_conf, 4)
            },
            "forensic_signal_metrics": forensic_metrics,
            "latency_ms": inference_latency_ms
        }
        
        if return_embedding:
            emb = outputs["forensic_embedding"][0].cpu().numpy().tolist()
            result["forensic_embedding_256d"] = emb
            
        return result


def main():
    parser = argparse.ArgumentParser(description="VoiceGuard Production Real-Time Threat Inspection")
    parser.add_argument("--audio", type=str, default=None, help="Path to input audio file (WAV/FLAC/OGG/MP3)")
    parser.add_argument("--checkpoint", type=str, default="D:/voiceguard/checkpoints/best_voiceguard_wav2vec2.pt", help="Path to checkpoint")
    parser.add_argument("--threshold", type=float, default=0.75, help="Threat kill-switch threshold (0.0 to 1.0)")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature scaling factor")
    args = parser.parse_args()

    engine = VoiceGuardInferenceEngine(
        checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None,
        threshold=args.threshold,
        temperature=args.temp
    )

    if args.audio and os.path.exists(args.audio):
        logger.info(f"Analyzing audio file: {args.audio}")
        res = engine.inspect_audio(args.audio)
    else:
        logger.info("No audio file supplied; generating synthetic audio sample for demo inference...")
        t = np.linspace(0, 3.0, 48000, endpoint=False)
        dummy_audio = (np.sin(2 * np.pi * 180.0 * t) + 0.3 * np.random.normal(0, 0.1, 48000)).astype(np.float32)
        res = engine.inspect_audio(dummy_audio)

    print("\n" + "=" * 70)
    print("VOICEGUARD REAL-TIME THREAT BIOMETRICS REPORT (SIH26104)")
    print("=" * 70)
    print(json.dumps(res, indent=4))
    print("=" * 70)


if __name__ == "__main__":
    main()