"""
VoiceGuard Training & Optimization Protocol (SIH26104)
Problem Statement: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

Training Specifications:
1. Dual-Track Datasets:
   - English: ASVspoof 2019/2021 LA + LibriSpeech
   - Indic: IndicSuperb / IndicSynth (Hindi & Tamil bona fide and cloned speech)
2. Multi-Task Combined Loss:
   L_total = L_AMSoftmax(binary, s=30.0, m=0.35) + 0.3 * L_CE(vocoder_subtypes)
3. Differential Learning Rates:
   - 1e-5 for unfrozen Wav2Vec2 transformer layers
   - 1e-4 for classification heads & forensic projector
4. Optimizer & Scheduler:
   - AdamW (beta1=0.9, beta2=0.98, weight_decay=1e-4)
   - Linear warmup (10% steps) + Cosine Annealing decay
5. Evaluation Metric:
   - Equal Error Rate (EER) computed on validation set; early stopping with best checkpoint tracking.
"""

import os
import math
import time
import argparse
import logging
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

from voiceguard_wav2vec2 import VoiceGuardWav2Vec2, AMSoftmaxLoss
from dataset import VoiceGuardDataset, get_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("VoiceGuard.Train")


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and the optimal decision threshold.
    Args:
        scores: Probability or score indicating spoof (higher score = more likely spoof/fake).
        labels: Ground truth binary labels (0 = genuine/bona fide, 1 = spoof/fake).
    Returns:
        eer: Equal Error Rate in percentage (0 to 100%).
        threshold: Decision threshold where False Positive Rate == False Negative Rate.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    
    # Locate point where FPR and FNR intersect
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0 * 100.0)
    best_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, best_threshold


class VoiceGuardTrainer:
    """Production Trainer implementing SIH26104 VoiceGuard optimization protocol."""
    def __init__(
        self,
        model: VoiceGuardWav2Vec2,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str = "D:/voiceguard/checkpoints",
        lr_backbone: float = 1e-5,
        lr_heads: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 10,
        warmup_ratio: float = 0.10,
        vocoder_loss_weight: float = 0.30,
        am_scale: float = 30.0,
        am_margin: float = 0.35,
        patience: int = 3
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.epochs = epochs
        self.vocoder_loss_weight = vocoder_loss_weight
        self.patience = patience
        os.makedirs(output_dir, exist_ok=True)

        # 1. Dual Loss Functions
        self.am_loss_fn = AMSoftmaxLoss(in_features=256, num_classes=2, scale=am_scale, margin=am_margin).to(device)
        self.ce_vocoder_fn = nn.CrossEntropyLoss().to(device)

        # 2. Differential Learning Rates
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "wav2vec2" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        # Also include AM-Softmax learnable weights in heads group
        head_params.extend(list(self.am_loss_fn.parameters()))

        optimizer_grouped_parameters = [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_heads}
        ]

        # 3. AdamW Optimizer
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.98),
            weight_decay=weight_decay
        )

        # 4. Learning Rate Scheduler (Warmup + Cosine Annealing)
        total_steps = len(train_loader) * epochs
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        self.best_val_eer = float("inf")
        self.best_checkpoint_path = os.path.join(output_dir, "best_voiceguard_wav2vec2.pt")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_am_loss = 0.0
        total_voc_loss = 0.0
        num_batches = len(self.train_loader)

        start_time = time.time()
        for batch_idx, batch in enumerate(self.train_loader):
            waveforms = batch["waveform"].to(self.device)
            binary_labels = batch["binary_label"].to(self.device)
            vocoder_labels = batch["vocoder_label"].to(self.device)

            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(waveforms)
            
            # L_total = L_am(binary) + 0.3 * L_ce(vocoder)
            loss_am = self.am_loss_fn(outputs["forensic_embedding"], binary_labels)
            loss_voc = self.ce_vocoder_fn(outputs["vocoder_logits"], vocoder_labels)
            loss = loss_am + (self.vocoder_loss_weight * loss_voc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_am_loss += loss_am.item()
            total_voc_loss += loss_voc.item()

            if (batch_idx + 1) % max(1, num_batches // 5) == 0 or (batch_idx + 1) == num_batches:
                logger.info(
                    f"Epoch [{epoch}/{self.epochs}] Step [{batch_idx+1}/{num_batches}] "
                    f"Loss: {loss.item():.4f} (AM: {loss_am.item():.4f}, Voc: {loss_voc.item():.4f}) "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )

        elapsed = time.time() - start_time
        return {
            "loss": total_loss / max(1, num_batches),
            "am_loss": total_am_loss / max(1, num_batches),
            "voc_loss": total_voc_loss / max(1, num_batches),
            "time_sec": elapsed
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Runs validation and computes Equal Error Rate (EER)."""
        self.model.eval()
        scores = []
        ground_truths = []
        voc_correct = 0
        voc_total = 0

        for batch in self.val_loader:
            waveforms = batch["waveform"].to(self.device)
            binary_labels = batch["binary_label"].to(self.device)
            vocoder_labels = batch["vocoder_label"].to(self.device)

            outputs = self.model(waveforms)
            probs = F.softmax(outputs["binary_logits"], dim=-1)
            spoof_prob = probs[:, 1].cpu().numpy()

            scores.extend(spoof_prob.tolist())
            ground_truths.extend(binary_labels.cpu().numpy().tolist())

            voc_preds = torch.argmax(outputs["vocoder_logits"], dim=-1)
            voc_correct += (voc_preds == vocoder_labels).sum().item()
            voc_total += len(vocoder_labels)

        scores_arr = np.array(scores)
        gt_arr = np.array(ground_truths)
        
        # Check if both classes are present
        if len(np.unique(gt_arr)) > 1:
            eer, threshold = compute_eer(scores_arr, gt_arr)
        else:
            eer, threshold = 0.0, 0.5

        voc_acc = (voc_correct / max(1, voc_total)) * 100.0
        return {
            "val_eer": round(eer, 3),
            "val_threshold": round(threshold, 4),
            "val_vocoder_acc": round(voc_acc, 2)
        }

    def train(self):
        """Full training loop with early stopping."""
        logger.info(f"Starting VoiceGuard Training ({self.epochs} epochs)...")
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            logger.info(
                f"[Epoch {epoch} Summary] "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val EER: {val_metrics['val_eer']:.2f}% | "
                f"Val Vocoder Acc: {val_metrics['val_vocoder_acc']:.2f}% | "
                f"Threshold: {val_metrics['val_threshold']:.4f} "
                f"({train_metrics['time_sec']:.1f}s)"
            )

            # Checkpoint best model on validation EER
            if val_metrics["val_eer"] < self.best_val_eer:
                self.best_val_eer = val_metrics["val_eer"]
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "am_softmax_state_dict": self.am_loss_fn.state_dict(),
                    "best_eer": self.best_val_eer,
                    "threshold": val_metrics["val_threshold"],
                    "config": self.model.wav2vec2.config.to_dict()
                }, self.best_checkpoint_path)
                logger.info(f"==> Saved new best model checkpoint to {self.best_checkpoint_path} (EER: {self.best_val_eer:.2f}%)")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered after {self.patience} epochs without EER improvement.")
                    break

        logger.info(f"Training Complete. Best Validation EER: {self.best_val_eer:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="VoiceGuard SIH26104 Training Pipeline")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_path", type=str, default="D:/ML model VG" if os.path.isdir("D:/ML model VG") else "facebook/wav2vec2-base")
    parser.add_argument("--output_dir", type=str, default="D:/voiceguard/checkpoints")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device} | Model Backbone: {args.model_path}")

    # 1. Initialize Model
    model = VoiceGuardWav2Vec2(
        pretrained_model_name=args.model_path,
        num_vocoder_classes=5,
        unfreeze_top_n_layers=3
    )

    # 2. DataLoaders (with synthetic fallback for immediate verification)
    train_loader = get_dataloader(batch_size=args.batch_size, is_training=True)
    val_loader = get_dataloader(batch_size=args.batch_size, is_training=False)

    # 3. Train
    trainer = VoiceGuardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        epochs=args.epochs
    )
    trainer.train()


if __name__ == "__main__":
    main()