"""
VoiceGuard Dataset & Data Loading Pipeline
Problem Statement SIH26104: Dual-Track Indic & English Speech Biometrics

Tracks:
1. English Track: ASVspoof 2019 LA / 2021 LA (spoof samples) + LibriSpeech (bona fide).
2. Indic Track: IndicSuperb / IndicSynth (Hindi and Tamil bona fide & cloned voice datasets).

Includes:
- Robust audio reading & 16 kHz resampling.
- Integration with LossyAudioAugmenter (anti-lossy codec degradation).
- Vocoder subtype label mapping: [bona_fide, diffwave, melgan, wavenet, elevenlabs_neural].
- Zero-dependency Synthetic Audio Fallback Generator for offline testing and self-verification.
"""

import os
import glob
import random
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as F_audio

from voiceguard_wav2vec2 import LossyAudioAugmenter

logger = logging.getLogger("VoiceGuard.Dataset")

VOCODER_TO_IDX = {
    "bona_fide": 0,
    "diffwave": 1,
    "melgan": 2,
    "wavenet": 3,
    "elevenlabs_neural": 4
}


class VoiceGuardDataset(Dataset):
    """
    Dual-Track Dataset for Voice Cloning Detection.
    Loads audio, resamples to 16 kHz, applies on-the-fly lossy augmentation,
    and returns 3.0s windowed tensors along with binary threat labels and vocoder subtype labels.
    """
    def __init__(
        self,
        samples: Optional[List[Dict[str, Any]]] = None,
        manifest_path: Optional[str] = None,
        is_training: bool = True,
        target_sample_rate: int = 16000,
        target_length_samples: int = 48000,
        use_augmentation: bool = True,
        num_synthetic_samples_if_empty: int = 100
    ):
        super().__init__()
        self.is_training = is_training
        self.target_sample_rate = target_sample_rate
        self.target_length_samples = target_length_samples
        self.augmenter = LossyAudioAugmenter(
            sample_rate=target_sample_rate,
            target_length_samples=target_length_samples
        ) if use_augmentation else None

        self.samples = []
        if samples is not None:
            self.samples.extend(samples)
        elif manifest_path and os.path.exists(manifest_path):
            self._load_manifest(manifest_path)

        # Fallback to deterministic synthetic samples if no manifest or samples provided
        if len(self.samples) == 0:
            logger.info(f"No audio files provided; generating {num_synthetic_samples_if_empty} synthetic samples for smoke testing.")
            self._generate_synthetic_samples(num_synthetic_samples_if_empty)

    def _load_manifest(self, manifest_path: str):
        """
        Parses ASVspoof-style or TSV protocol manifest:
        Format: file_path \t binary_label (0=bonafide, 1=spoof) \t vocoder_type (e.g. diffwave) \t language
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    path = parts[0]
                    lbl = parts[1].lower()
                    binary_label = 0 if (lbl in ["bonafide", "bona_fide", "0", "real"]) else 1
                    vocoder_str = parts[2].lower() if len(parts) >= 3 else ("bona_fide" if binary_label == 0 else "melgan")
                    vocoder_label = VOCODER_TO_IDX.get(vocoder_str, 1 if binary_label == 1 else 0)
                    lang = parts[3].lower() if len(parts) >= 4 else "en"
                    
                    self.samples.append({
                        "file_path": path,
                        "binary_label": binary_label,
                        "vocoder_label": vocoder_label,
                        "language": lang
                    })

    def _generate_synthetic_samples(self, n: int):
        """Generates synthetic harmonic and noise waveforms for fast validation."""
        for i in range(n):
            is_spoof = (i % 2 == 1)
            vocoder_idx = random.randint(1, 4) if is_spoof else 0
            lang = "hi" if (i % 3 == 0) else ("ta" if i % 3 == 1 else "en")
            self.samples.append({
                "synthetic": True,
                "is_spoof": is_spoof,
                "binary_label": 1 if is_spoof else 0,
                "vocoder_label": vocoder_idx,
                "language": lang,
                "id": i
            })

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, sample_info: Dict[str, Any]) -> torch.Tensor:
        """Loads audio file or creates synthetic speech-like signal."""
        if sample_info.get("synthetic", False):
            sr = self.target_sample_rate
            duration_s = 3.5
            t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
            f0 = np.random.uniform(120.0, 240.0)
            
            signal = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t) + 0.25 * np.sin(2 * np.pi * 3 * f0 * t)
            
            if sample_info["is_spoof"]:
                jitter = np.random.normal(0, 0.05, len(t))
                signal += 0.3 * np.sin(2 * np.pi * 4000.0 * (t + jitter))
                
            waveform = torch.from_numpy(signal).float().unsqueeze(0)
            return waveform

        path = sample_info["file_path"]
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.target_sample_rate:
                waveform = F_audio.resample(waveform, sr, self.target_sample_rate)
            return waveform
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}. Returning zeros.")
            return torch.zeros(1, self.target_length_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        waveform = self._load_audio(sample_info)
        
        if self.augmenter is not None:
            waveform = self.augmenter(waveform, training=self.is_training)
        else:
            if waveform.size(-1) < self.target_length_samples:
                waveform = torch.nn.functional.pad(waveform, (0, self.target_length_samples - waveform.size(-1)))
            else:
                waveform = waveform[..., :self.target_length_samples]

        return {
            "waveform": waveform.squeeze(0),
            "binary_label": torch.tensor(sample_info["binary_label"], dtype=torch.long),
            "vocoder_label": torch.tensor(sample_info["vocoder_label"], dtype=torch.long)
        }


def get_dataloader(
    samples: Optional[List[Dict[str, Any]]] = None,
    manifest_path: Optional[str] = None,
    batch_size: int = 8,
    is_training: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Utility function to instantiate a DataLoader."""
    dataset = VoiceGuardDataset(
        samples=samples,
        manifest_path=manifest_path,
        is_training=is_training
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    print("Testing VoiceGuard Dataset Pipeline...")
    loader = get_dataloader(batch_size=4, is_training=True)
    batch = next(iter(loader))
    print("Batch Waveform Shape:    ", tuple(batch["waveform"].shape))
    print("Batch Binary Labels:     ", batch["binary_label"].tolist())
    print("Batch Vocoder Labels:    ", batch["vocoder_label"].tolist())
    print("Dataset verification successful!")