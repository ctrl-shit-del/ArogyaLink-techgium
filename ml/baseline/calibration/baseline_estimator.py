"""
Synera 2.0 — Baseline Estimator
Computes running mean and standard deviation from a stream of readings
using Welford's online algorithm (numerically stable, no full history needed).
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class BaselineEstimator:
    """
    Online mean + std estimator using Welford's algorithm.
    Processes one (5,) feature vector at a time — no accumulation of raw data.
    """

    def __init__(self, n_features: int = 5):
        self.n_features = n_features
        self._n     = 0
        self._mean  = np.zeros(n_features, dtype=np.float64)
        self._M2    = np.zeros(n_features, dtype=np.float64)  # sum of squared deviations

    def update(self, x: np.ndarray) -> None:
        """
        Incorporate one observation.
        x: (n_features,) float array
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.n_features,):
            raise ValueError(f"Expected ({self.n_features},), got {x.shape}")
        self._n += 1
        delta   = x - self._mean
        self._mean += delta / self._n
        delta2  = x - self._mean
        self._M2 += delta * delta2

    @property
    def count(self) -> int:
        return self._n

    @property
    def mean(self) -> np.ndarray:
        """Current running mean. Shape: (n_features,)"""
        return self._mean.copy()

    @property
    def variance(self) -> np.ndarray:
        """Unbiased variance (n≥2 required). Shape: (n_features,)"""
        if self._n < 2:
            return np.zeros(self.n_features, dtype=np.float64)
        return self._M2 / (self._n - 1)

    @property
    def std(self) -> np.ndarray:
        """Standard deviation. Shape: (n_features,)"""
        return np.sqrt(self.variance)

    def coefficient_of_variation(self) -> np.ndarray:
        """
        CV = std / |mean|.  Used to assess baseline stability.
        Low CV → stable baseline → high quality calibration.
        Returns (n_features,) float.
        """
        m = np.abs(self._mean)
        s = self.std
        safe_m = np.where(m > 1e-9, m, 1e-9)
        return s / safe_m

    def reset(self) -> None:
        self._n    = 0
        self._mean = np.zeros(self.n_features, dtype=np.float64)
        self._M2   = np.zeros(self.n_features, dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "n":    self._n,
            "mean": self._mean.tolist(),
            "M2":   self._M2.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineEstimator":
        n_features = len(d["mean"])
        est = cls(n_features)
        est._n    = d["n"]
        est._mean = np.array(d["mean"], dtype=np.float64)
        est._M2   = np.array(d["M2"],   dtype=np.float64)
        return est
