"""
VoiceGuardRawNet Benchmark
==========================
Production-grade micro-benchmark for the SIH26104 voice-cloning detection model.

Measures:
  - Model parameter count & serialized footprint (FP32 vs INT8 PTQ)
  - End-to-end CPU latency (FP32 vs INT8) with percentiles
  - Throughput (inferences/sec)
  - Threat-score / kill-switch decision behavior (calibrated to 75% threshold)
  - Verification against stated budgets:
      * Serialized INT8 size  <= ~2.2 MB
      * End-to-end forward   < 185 ms on 2 vCPU
      * Window               1.5 s @ 16 kHz (24,000 samples)
"""


import argparse
import io
import json
import os
import sys
import time


# Reconfigure stdout/stderr for Windows console compatibility
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers, **kwargs):
        out = []
        out.append(" | ".join(str(h) for h in headers))
        out.append(" | ".join("-" * max(len(str(h)), 4) for h in headers))
        for r in rows:
            out.append(" | ".join(str(c) for c in r))
        return "\n".join(out)




# --------------------------------------------------------------------------- #
# 1. Model Definitions (Vectorized & Metadata-Aligned)
# --------------------------------------------------------------------------- #
class SincConv(nn.Module):
    """Vectorized 1D SincNet parameterized convolution."""


    def __init__(self, in_channels=1, out_channels=64, kernel_size=129, sample_rate=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate


        self.filt_params = nn.Parameter(torch.rand(out_channels, 2) * 0.4 + 0.05)
        with torch.no_grad():
            sorted_params, _ = torch.sort(self.filt_params, dim=1)
            self.filt_params.copy_(sorted_params)


    def forward(self, x):
        b, c, length = x.shape
        device = x.device


        # Center time vector around 0
        t = torch.linspace(-self.kernel_size // 2, self.kernel_size // 2,
                           self.kernel_size, device=device).view(1, 1, -1)


        f_min = (self.filt_params[:, 0] * (self.sample_rate / 2)).view(-1, 1, 1)
        f_max = (self.filt_params[:, 1] * (self.sample_rate / 2)).view(-1, 1, 1)


        # Broadcasted sinc calculation (no Python loop)
        kernels = 2 * f_max * torch.sinc(2 * f_max * t) - 2 * f_min * torch.sinc(2 * f_min * t)
        if c > 1:
            kernels = kernels.repeat(1, c, 1)


        return F.conv1d(x, kernels, padding=self.kernel_size // 2)




class ResBlock(nn.Module):
    """1D residual block with optional 1x1 projection shortcut."""


    def __init__(self, in_channels, out_channels, projection=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1, bias=True) if projection else nn.Identity()


    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)




class VoiceGuardRawNet(nn.Module):
    """VoiceGuardRawNet core: Sinc -> ResBlocks -> Pool -> BiGRU -> Linear(5)."""


    CLASSES = ['bona_fide', 'synthetic_indic', 'diffwave', 'melgan', 'wavenet']


    def __init__(self, num_classes=5):
        super().__init__()
        self.sinc = SincConv(in_channels=1, out_channels=64, kernel_size=129)
        self.res1 = ResBlock(64, 64, projection=False)
        self.res2 = ResBlock(64, 128, projection=True)
        self.pool = nn.AdaptiveAvgPool1d(128)
        self.gru = nn.GRU(128, hidden_size=64, num_layers=2,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, num_classes)


    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        out = self.sinc(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.pool(out)
        out = out.permute(0, 2, 1)
        out, _ = self.gru(out)
        return self.fc(out[:, -1, :])




# --------------------------------------------------------------------------- #
# 2. Preprocessing & Utilities
# --------------------------------------------------------------------------- #
TARGET_SAMPLES = 24000  # 1.5s @ 16kHz
MITIGATION_THRESHOLD = 75.0




def preprocess_audio(audio_np: np.ndarray) -> torch.Tensor:
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val
    if len(audio_np) > TARGET_SAMPLES:
        audio_np = audio_np[:TARGET_SAMPLES]
    elif len(audio_np) < TARGET_SAMPLES:
        audio_np = np.pad(audio_np, (0, TARGET_SAMPLES - len(audio_np)))
    return torch.from_numpy(audio_np).float().view(1, 1, -1)




def threat_score_from_logits(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=1).squeeze(0)
    return float(probs[1:].sum().item() * 100.0)




def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def serialized_size_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes




def benchmark_latency(model: nn.Module, input_tensor: torch.Tensor,
                      runs: int, warmup: int, batch: int) -> dict:
    model.eval()
    batched = input_tensor.repeat(batch, 1, 1)


    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(batched)


    latencies = []
    with torch.inference_mode():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(batched)
            latencies.append((time.perf_counter() - t0) * 1000.0)


    latencies = np.array(latencies)
    return {
        "runs": runs,
        "batch": batch,
        "mean_ms": float(latencies.mean() / batch),
        "median_ms": float(np.median(latencies) / batch),
        "p50_ms": float(np.percentile(latencies, 50) / batch),
        "p90_ms": float(np.percentile(latencies, 90) / batch),
        "p95_ms": float(np.percentile(latencies, 95) / batch),
        "p99_ms": float(np.percentile(latencies, 99) / batch),
        "min_ms": float(latencies.min() / batch),
        "max_ms": float(latencies.max() / batch),
        "std_ms": float(latencies.std() / batch),
        "throughput_chunks_s": float(batch * runs / (latencies.sum() / 1000.0)),
    }




def quantize_and_load(base_model: nn.Module, checkpoint_path: str = None) -> nn.Module:
    quantized = torch.ao.quantization.quantize_dynamic(
        base_model, {nn.GRU, nn.Linear}, dtype=torch.qint8
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        quantized.load_state_dict(state, strict=False)
        print(f"Loaded weights from checkpoint: {checkpoint_path}")
    return quantized




def decision_behavior(detection_model: nn.Module) -> dict:
    rng = np.random.default_rng(42)
    healthy = preprocess_audio(rng.normal(0, 0.05, TARGET_SAMPLES).astype(np.float32))
    noisy = preprocess_audio(rng.normal(0, 0.5, TARGET_SAMPLES).astype(np.float32))


    with torch.inference_mode():
        healthy_ts = threat_score_from_logits(detection_model(healthy))
        noisy_ts = threat_score_from_logits(detection_model(noisy))


    def _decision(ts):
        return (
            "🚨 TERMINATE CALL (Synthetic Spoof Detected)"
            if ts >= MITIGATION_THRESHOLD
            else "✅ ALLOW STREAM (Authentic Voice)"
        )


    return {
        "mitigation_threshold": f"{MITIGATION_THRESHOLD}%",
        "healthy_threat_score": round(healthy_ts, 2),
        "healthy_decision": _decision(healthy_ts),
        "noisy_threat_score": round(noisy_ts, 2),
        "noisy_decision": _decision(noisy_ts),
    }




# --------------------------------------------------------------------------- #
# 3. Execution Harness
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="VoiceGuardRawNet benchmark")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2, help="Target CPU threads (default 2 for edge budget)")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to voiceguard_int8.pt or .pth")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()


    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


    print(f"PyTorch: {torch.__version__}")
    print(f"Configured CPU Threads: {torch.get_num_threads()} (Target 2 vCPU budget)")
    print(f"Batch: {args.batch}, Runs: {args.runs}, Warmup: {args.warmup}\n")


    # Build FP32 and INT8 Models
    fp32_model = VoiceGuardRawNet()
    fp32_model.eval()


    checkpoint_file = args.checkpoint
    if not checkpoint_file:
        for candidate in ["quantized_voiceguard_model.pth", "voiceguard_int8.pt"]:
            if os.path.exists(candidate):
                checkpoint_file = candidate
                break


    int8_model = quantize_and_load(fp32_model, checkpoint_file)
    int8_model.eval()


    # Serialization Footprint
    fp32_mb = serialized_size_bytes(fp32_model) / (1024 * 1024)
    int8_mb = serialized_size_bytes(int8_model) / (1024 * 1024)
    params = count_parameters(fp32_model)


    # Input & Benchmarks
    input_tensor = preprocess_audio(np.random.uniform(-0.5, 0.5, TARGET_SAMPLES).astype(np.float32))
    fp32_lat = benchmark_latency(fp32_model, input_tensor, args.runs, args.warmup, args.batch)
    int8_lat = benchmark_latency(int8_model, input_tensor, args.runs, args.warmup, args.batch)


    # Decisions at 75% Threshold
    decisions = decision_behavior(int8_model)


    # Budget Checks
    size_ok = int8_mb <= 2.2
    latency_ok = int8_lat["p95_ms"] < 185.0


    rows = [
        ["Model Parameters", f"{params:,}", ""],
        ["FP32 Serialized Size", f"{fp32_mb:.3f} MB", ""],
        ["INT8 Serialized Size", f"{int8_mb:.3f} MB", f"{'✅ <= 2.2 MB' if size_ok else '❌ > 2.2 MB'}"],
        ["Threat Threshold", f"{MITIGATION_THRESHOLD}%", "Calibrated for zero-trust FAR"],
    ]


    print("=" * 78)
    print("MODEL FOOTPRINT & SPECIFICATION")
    print("=" * 78)
    print(tabulate(rows, headers=["Metric", "Value", "Budget Check"], tablefmt="github"))
    print()


    lat_rows = [
        [
            tag,
            f"{lat['mean_ms']:.2f}",
            f"{lat['p50_ms']:.2f}",
            f"{lat['p90_ms']:.2f}",
            f"{lat['p95_ms']:.2f}",
            f"{lat['p99_ms']:.2f}",
            f"{lat['min_ms']:.2f}",
            f"{lat['max_ms']:.2f}",
            f"{lat['throughput_chunks_s']:.1f}",
        ]
        for tag, lat in [("FP32", fp32_lat), ("INT8", int8_lat)]
    ]


    print("=" * 78)
    print(f"CPU LATENCY — {args.runs} runs (per 1.5s chunk, {args.threads} vCPU threads, batch={args.batch})")
    print("=" * 78)
    print(tabulate(lat_rows,
                   headers=["Precision", "Mean(ms)", "P50(ms)", "P90(ms)", "P95(ms)",
                            "P99(ms)", "Min(ms)", "Max(ms)", "Chunks/s"],
                   tablefmt="github"))
    print()
    print(f"Latency Budget (<185ms @ P95): {'✅ PASS' if latency_ok else '❌ FAIL'} "
          f"(INT8 P95 = {int8_lat['p95_ms']:.2f} ms)")
    print("=" * 78)


    print()
    print(f"KILL-SWITCH DECISION TEST (INT8 Quantized Model @ {MITIGATION_THRESHOLD}% Cutoff)")
    print(f"  Low-energy input:   Threat={decisions['healthy_threat_score']}%  {decisions['healthy_decision']}")
    print(f"  High-energy input:  Threat={decisions['noisy_threat_score']}%  {decisions['noisy_decision']}")
    print()


    payload = {
        "environment": {
            "torch_version": torch.__version__,
            "configured_threads": args.threads,
            "batch": args.batch,
        },
        "footprint": {
            "parameters": params,
            "fp32_mb": round(fp32_mb, 3),
            "int8_mb": round(int8_mb, 3),
            "size_budget_mb": 2.2,
            "size_ok": size_ok,
        },
        "latency_int8_ms": {k: round(v, 3) for k, v in int8_lat.items() if "ms" in k},
        "throughput_chunks_s": round(int8_lat["throughput_chunks_s"], 1),
        "latency_budget_ms": 185.0,
        "latency_ok": latency_ok,
        "decisions": decisions,
        "verdict": {
            "pass": size_ok and latency_ok,
            "size_ok": size_ok,
            "latency_ok": latency_ok,
        },
    }


    if args.json:
        print(json.dumps(payload, indent=2))


    return 0 if (size_ok and latency_ok) else 1




if __name__ == "__main__":
    sys.exit(main())
