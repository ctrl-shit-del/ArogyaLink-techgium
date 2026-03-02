"""
Synera 2.0 — Vital Sign Window Dataset
PyTorch Dataset wrapper around (N_windows, 10, 5) numpy arrays.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class VitalWindowDataset(Dataset):
    """
    PyTorch Dataset for LSTM autoencoder training.

    Returns:
        (window_tensor, window_tensor) — same tensor twice (autoencoder target = input)
        label: anomaly label (0/1) — stored but NOT used during training;
               used only for post-training evaluation metrics.
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels:  np.ndarray | None = None,
    ):
        """
        Args:
            windows: (N, 10, 5) float32 — normalised vital windows
            labels:  (N,)       int32   — 0=normal, 1=anomalous (optional)
        """
        if windows.ndim != 3 or windows.shape[1:] != (10, 5):
            raise ValueError(f"Expected (N, 10, 5), got {windows.shape}")
        self.windows = torch.from_numpy(windows.astype(np.float32))
        self.labels  = (
            torch.from_numpy(labels.astype(np.int32)) if labels is not None
            else torch.zeros(len(windows), dtype=torch.int32)
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        window = self.windows[idx]
        label  = self.labels[idx]
        # Autoencoder: input == target
        return window, window, label

    @property
    def shape(self):
        return self.windows.shape
