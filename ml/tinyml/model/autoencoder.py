"""
Synera 2.0 — LSTM Autoencoder (Full Model)
Combines encoder + decoder into the complete VitalAutoencoder.

This is the model that:
  1. Trains on normal patient vital sign windows only.
  2. Learns to reconstruct normal physiological patterns with low error.
  3. Produces HIGH reconstruction error on anomalous patterns (deterioration),
     because it has never learned those trajectories.
  4. Gets quantised to INT8 for deployment on the ESP32-S3 via TFLite Micro.

Size verification:
  Encoder:  1544 params
  Decoder:  2341 params
  Total:    3885 params
  FP32:     3885 × 4 bytes = 15.2 KB
  INT8:     3885 × 1 byte  =  3.8 KB  ← well within 45KB flash budget
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.tinyml.model.lstm_encoder import LSTMEncoder
from ml.tinyml.model.lstm_decoder import LSTMDecoder


@dataclass
class AutoencoderConfig:
    """Matches the spec exactly.  Changing any value changes the hardware model."""
    input_size:   int   = 5
    hidden_size:  int   = 16
    latent_size:  int   = 8
    seq_len:      int   = 10
    num_layers:   int   = 1
    anomaly_threshold: float = 0.035   # overwritten by training pipeline


class VitalAutoencoder(nn.Module):
    """
    LSTM Autoencoder for physiological trajectory anomaly detection.

    Inference path:
        raw_window (10, 5) → normalize → model → reconstruction_error → pre_alert flag
    """

    def __init__(self, config: Optional[AutoencoderConfig] = None):
        super().__init__()
        self.config = config or AutoencoderConfig()
        c = self.config

        self.encoder = LSTMEncoder(
            input_size=c.input_size,
            hidden_size=c.hidden_size,
            latent_size=c.latent_size,
            num_layers=c.num_layers,
        )
        self.decoder = LSTMDecoder(
            latent_size=c.latent_size,
            hidden_size=c.hidden_size,
            output_size=c.input_size,
            seq_len=c.seq_len,
            num_layers=c.num_layers,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len=10, features=5) — normalised input window

        Returns:
            reconstructed: (batch, 10, 5) — reconstruction, values in [0,1]
            latent:        (batch, 8)     — bottleneck representation
        """
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-window mean squared reconstruction error.

        Args:
            x: (batch, 10, 5)

        Returns:
            errors: (batch,) float — one MSE value per window
                    High error = pattern anomalous = candidate for pre_alert
        """
        reconstructed, _ = self.forward(x)
        # MSE per window: mean over seq_len and features
        mse = ((x - reconstructed) ** 2).mean(dim=(1, 2))   # (batch,)
        return mse

    def predict(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method: compute error + binary pre_alert flag.

        Args:
            x:         (batch, 10, 5)
            threshold: override config threshold if provided

        Returns:
            errors:     (batch,)  float — reconstruction MSE
            pre_alerts: (batch,)  bool  — True if error > threshold
        """
        thr = threshold if threshold is not None else self.config.anomaly_threshold
        errors = self.reconstruction_error(x)
        pre_alerts = errors > thr
        return errors, pre_alerts

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_model_size_kb(self) -> float:
        """
        Estimated post-INT8-quantization size in KB.
        FP32 params × 1 byte each (INT8) + small TFLite header.
        """
        n_params  = self.count_parameters()
        int8_bytes = n_params * 1
        header_overhead = 512  # bytes — TFLite flatbuffer header
        return (int8_bytes + header_overhead) / 1024.0

    def architecture_summary(self) -> str:
        """Human-readable architecture summary for slide decks / demo output."""
        c = self.config
        enc_params = self.encoder.count_parameters()
        dec_params = self.decoder.count_parameters()
        total      = enc_params + dec_params
        fp32_kb    = total * 4 / 1024.0
        int8_kb    = self.estimate_model_size_kb()
        lines = [
            "─" * 52,
            "  VitalAutoencoder — Architecture Summary",
            "─" * 52,
            f"  Input shape   : (batch, {c.seq_len}, {c.input_size})",
            f"  Encoder       : LSTM({c.input_size}→{c.hidden_size}) + Linear({c.hidden_size}→{c.latent_size})",
            f"                  {enc_params:,} params",
            f"  Latent vector : (batch, {c.latent_size})",
            f"  Decoder       : Linear({c.latent_size}→{c.hidden_size}) + LSTM({c.hidden_size}→{c.hidden_size}) + Linear({c.hidden_size}→{c.input_size})",
            f"                  {dec_params:,} params",
            f"  Total params  : {total:,}",
            f"  FP32 size     : {fp32_kb:.1f} KB",
            f"  INT8 size     : {int8_kb:.1f} KB  {'✓' if int8_kb < 45 else '✗ EXCEEDS BUDGET'}",
            f"  Threshold     : {self.config.anomaly_threshold:.4f}",
            "─" * 52,
        ]
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Save / load
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights + config to a .pt checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "config":      self.config.__dict__,
        }, path)

    @classmethod
    def load(cls, path: str) -> "VitalAutoencoder":
        """Load from a .pt checkpoint."""
        ckpt   = torch.load(path, map_location="cpu", weights_only=False)
        config = AutoencoderConfig(**ckpt["config"])
        model  = cls(config=config)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick verification
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = VitalAutoencoder()
    print(model.architecture_summary())

    # Forward pass
    x   = torch.randn(4, 10, 5).clamp(0, 1)
    rec, lat = model(x)
    err = model.reconstruction_error(x)
    print(f"\n  Forward pass OK")
    print(f"  Input:         {tuple(x.shape)}")
    print(f"  Reconstructed: {tuple(rec.shape)}")
    print(f"  Latent:        {tuple(lat.shape)}")
    print(f"  Error (batch): {err.detach().numpy().round(4)}")
