"""
Synera 2.0 — Reconstruction Loss
MSE computed over (seq_len, features) — reports one scalar per batch item.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """
    Per-sample mean squared error summed/averaged over sequence and feature dims.

    Formula (per sample):
        L_i = mean_{t,f} (x_{i,t,f} - x̂_{i,t,f})²

    Batch loss returned is the mean over the batch dimension.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: "mean" (default) averages over batch.
                       "sum"  sums over batch.
                       "none" returns (batch,) tensor.
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean'|'sum'|'none', got {reduction!r}")
        self.reduction = reduction

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_hat: (batch, seq=10, features=5) — reconstruction
            x:     (batch, seq=10, features=5) — original input
        Returns:
            Scalar (or (batch,) if reduction="none")
        """
        # Per-sample MSE over (seq, feature) dims → shape (batch,)
        per_sample = torch.mean((x_hat - x) ** 2, dim=(1, 2))

        if self.reduction == "mean":
            return per_sample.mean()
        elif self.reduction == "sum":
            return per_sample.sum()
        else:  # "none"
            return per_sample

    def per_sample(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Convenience: always returns (batch,) tensor regardless of `reduction`."""
        return torch.mean((x_hat - x) ** 2, dim=(1, 2))
