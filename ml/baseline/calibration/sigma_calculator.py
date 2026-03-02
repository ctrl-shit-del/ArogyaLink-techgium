"""
Synera 2.0 — Sigma (Z-Score) Calculator
Computes per-feature z-score deviation from a patient's personal baseline.
This is the mathematical heart of Synera's personalisation claim.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class SigmaCalculator:
    """
    Given a patient's baseline mean and std, computes how many standard
    deviations any new reading deviates from their personal normal.

    Rule C trigger condition: deviation > 1.5σ AND second derivative growing
    """

    def __init__(
        self,
        baseline_mean: np.ndarray,
        baseline_std:  np.ndarray,
        min_std:       float = 0.01,   # floor to avoid division by zero on very stable features
    ):
        """
        Args:
            baseline_mean: (5,) personal mean from calibration
            baseline_std:  (5,) personal std  from calibration
            min_std:       minimum std floor (prevents 0-division on rock-stable features)
        """
        self.mean    = np.asarray(baseline_mean, dtype=np.float64)
        self.std     = np.maximum(np.asarray(baseline_std, dtype=np.float64), min_std)
        self.min_std = min_std

    def z_score(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute signed z-score for each feature.

        z[i] = (x[i] - mean[i]) / std[i]

        Positive = above personal normal
        Negative = below personal normal

        Returns: (5,) float64
        """
        obs = np.asarray(observation, dtype=np.float64)
        return (obs - self.mean) / self.std

    def max_deviation_sigma(self, observation: np.ndarray) -> float:
        """
        Maximum absolute z-score across all features.
        Used as the scalar 'how far from normal' metric in Rule C.
        """
        return float(np.max(np.abs(self.z_score(observation))))

    def is_significant_deviation(
        self,
        observation: np.ndarray,
        threshold_sigma: float = 2.5,
        feature_indices: Optional[list] = None,
    ) -> bool:
        """
        Returns True if any (or specified) feature deviates > threshold_sigma.

        Args:
            feature_indices: if None, check all 5 features.
                             if list, only check those indices (e.g. [0,1,3] = HR+SpO2+BP).
        """
        z = self.z_score(observation)
        if feature_indices is not None:
            z = z[feature_indices]
        return bool(np.any(np.abs(z) > threshold_sigma))

    def deviation_summary(self, observation: np.ndarray) -> dict:
        """
        Rich deviation report for logging and display.

        Returns:
            {
                "z_scores": {"heart_rate": float, "spo2": float, ...},
                "max_sigma": float,
                "deviating_features": [str, ...],   # features with |z| > 1.5
                "critical_features":  [str, ...],   # features with |z| > 2.5
            }
        """
        from ml.shared.constants import FEATURE_NAMES, BASELINE_DEVIATION_SIGMA
        z = self.z_score(observation)
        z_dict = {name: round(float(z[i]), 3) for i, name in enumerate(FEATURE_NAMES)}
        return {
            "z_scores":           z_dict,
            "max_sigma":          round(float(np.max(np.abs(z))), 3),
            "deviating_features": [n for n, v in z_dict.items() if abs(v) > BASELINE_DEVIATION_SIGMA],
            "critical_features":  [n for n, v in z_dict.items() if abs(v) > 2.5],
        }

    def update(
        self,
        observation: np.ndarray,
        alpha: float = 0.001,
    ) -> None:
        """
        Exponential moving average update of the baseline.
        Very slow drift (alpha=0.001) absorbs gradual physiological change
        (e.g. patient improving over several days).
        Does NOT absorb acute deterioration (too slow).
        """
        obs = np.asarray(observation, dtype=np.float64)
        self.mean = (1.0 - alpha) * self.mean + alpha * obs
        # Also update std towards current deviation
        new_std = np.abs(obs - self.mean)
        self.std = np.maximum(
            (1.0 - alpha) * self.std + alpha * new_std,
            self.min_std,
        )
