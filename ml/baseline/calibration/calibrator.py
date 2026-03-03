"""
Synera 2.0 — Patient Calibrator
Personalised baseline learning engine.

The calibrator collects the first 30 minutes of sensor data from a patient
and computes their personal vital sign baseline.  After calibration, all
readings are evaluated as z-scores from this personal baseline — not from
a population average.

Demo talking point:
    "A 60-year-old hypertensive patient with a resting HR of 90 BPM is not
     alarming at 95 BPM.  A 25-year-old athlete with a resting HR of 52 BPM
     is significantly elevated at 95 BPM (+2.15σ).  Synera knows the
     difference because it learned from the patient, not from a textbook."
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.baseline.calibration.baseline_estimator import BaselineEstimator
from ml.baseline.calibration.sigma_calculator import SigmaCalculator
from ml.shared.constants import (
    CALIBRATION_WINDOW_READINGS,
    CALIBRATION_MIN_READINGS,
    ROLLING_UPDATE_ALPHA,
    FEATURE_NAMES,
    N_FEATURES,
)


class PatientCalibrator:
    """
    Manages the personalised baseline lifecycle for one patient:
      1. ACCUMULATION   — collecting readings until calibration window fills
      2. CALIBRATING    — enough readings; computing stable baseline
      3. CALIBRATED     — baseline established; all future readings scored
      4. ROLLING UPDATE — slow EMA drift to track multi-day physiology changes
    """

    def __init__(
        self,
        patient_id:  str,
        window_size: int = CALIBRATION_WINDOW_READINGS,
    ):
        self.patient_id        = patient_id
        self.window_size       = window_size
        self._estimator        = BaselineEstimator(N_FEATURES)
        self._sigma_calc: Optional[SigmaCalculator] = None
        self.is_calibrated     = False
        self.calibration_quality = 0.0   # 0–1

    # ─────────────────────────────────────────────────────────────────────────

    def add_reading(self, feature_vector: np.ndarray) -> dict:
        """
        Add one normalised feature vector [hr, spo2, temp, bp, motion].
        Updates the running estimator and checks if calibration is complete.

        Returns:
            {
                "is_calibrated": bool,
                "progress_pct":  float,     # 0–100
                "quality_score": float,     # 0–1
                "readings_collected": int
            }
        """
        vec = np.asarray(feature_vector, dtype=np.float64)
        if vec.shape != (N_FEATURES,):
            raise ValueError(f"Expected ({N_FEATURES},), got {vec.shape}")

        # During calibration skip motion feature for mean/std estimation
        # (motion is too bursty; we don't want motion spikes polluting the baseline)
        self._estimator.update(vec)

        n = self._estimator.count

        # Check for calibration completion
        if not self.is_calibrated:
            if n >= CALIBRATION_MIN_READINGS:
                quality = self._compute_quality()
                if n >= self.window_size or quality >= 0.80:
                    self._finalise_calibration()

        # After calibration, apply rolling update
        elif self._sigma_calc is not None:
            self._sigma_calc.update(vec, alpha=ROLLING_UPDATE_ALPHA)

        progress = min(100.0, n / self.window_size * 100.0)
        return {
            "is_calibrated":     self.is_calibrated,
            "progress_pct":      round(progress, 1),
            "quality_score":     round(self.calibration_quality, 3),
            "readings_collected": n,
        }

    def calibration_progress(self) -> dict:
        """Return current calibration status without adding a reading."""
        n        = self._estimator.count
        progress = min(100.0, n / self.window_size * 100.0)
        return {
            "is_calibrated":      self.is_calibrated,
            "progress_pct":       round(progress, 1),
            "quality_score":      round(self.calibration_quality, 3),
            "readings_collected": n,
        }

    def get_deviation(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Returns z-score deviation from personal baseline for each feature.
        Shape: (5,).  Requires calibration to be complete.
        Raises RuntimeError if not yet calibrated.
        """
        if not self.is_calibrated or self._sigma_calc is None:
            raise RuntimeError(
                f"Patient {self.patient_id} is not yet calibrated "
                f"({self._estimator.count}/{self.window_size} readings collected)."
            )
        return self._sigma_calc.z_score(feature_vector)

    def get_max_sigma(self, feature_vector: np.ndarray) -> float:
        """Max absolute z-score across all features — the scalar deviation signal."""
        return float(np.max(np.abs(self.get_deviation(feature_vector))))

    def is_significant(
        self,
        feature_vector: np.ndarray,
        threshold: float = 2.5,
    ) -> bool:
        """True if any feature deviates > threshold σ from personal baseline."""
        if not self.is_calibrated:
            return False
        return self._sigma_calc.is_significant_deviation(feature_vector, threshold)

    def deviation_report(self, feature_vector: np.ndarray) -> dict:
        """Full deviation breakdown per feature for logging and display."""
        if not self.is_calibrated:
            return {"error": "not_calibrated", "progress": self._estimator.count}
        return self._sigma_calc.deviation_summary(feature_vector)

    def update_rolling(self, feature_vector: np.ndarray) -> None:
        """
        Manually trigger a slow rolling update of the baseline.
        Normally called automatically by add_reading() post-calibration.
        """
        if self._sigma_calc is not None:
            self._sigma_calc.update(
                np.asarray(feature_vector, dtype=np.float64),
                alpha=ROLLING_UPDATE_ALPHA,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize the calibrated baseline to JSON for session persistence."""
        payload = {
            "patient_id":           self.patient_id,
            "window_size":          self.window_size,
            "is_calibrated":        self.is_calibrated,
            "calibration_quality":  self.calibration_quality,
            "estimator":            self._estimator.to_dict(),
            "baseline_mean":        (self._sigma_calc.mean.tolist() if self._sigma_calc else None),
            "baseline_std":         (self._sigma_calc.std.tolist()  if self._sigma_calc else None),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PatientCalibrator":
        """Load a persisted calibrator.  Returning patients skip recalibration."""
        with open(path, "r") as f:
            d = json.load(f)
        cal = cls(patient_id=d["patient_id"], window_size=d["window_size"])
        cal._estimator         = BaselineEstimator.from_dict(d["estimator"])
        cal.is_calibrated      = d["is_calibrated"]
        cal.calibration_quality = d["calibration_quality"]
        if d["is_calibrated"] and d["baseline_mean"]:
            cal._sigma_calc = SigmaCalculator(
                np.array(d["baseline_mean"]),
                np.array(d["baseline_std"]),
            )
        return cal

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def baseline_mean(self) -> Optional[np.ndarray]:
        return self._sigma_calc.mean.astype(np.float32) if self._sigma_calc else None

    @property
    def baseline_std(self) -> Optional[np.ndarray]:
        return self._sigma_calc.std.astype(np.float32) if self._sigma_calc else None

    def get_calibration_quality(self) -> float:
        """
        Quality metric based on coefficient of variation.
        The lower the CV (relative noise), the more reliable the baseline.
        Returns 0–1 where 1 = perfectly stable calibration window.
        """
        return self.calibration_quality

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_quality(self) -> float:
        """
        Quality = 1 - mean(CV across primary features HR, SpO2, Temp).
        High CV → noisy calibration window → low quality.
        """
        cv = self._estimator.coefficient_of_variation()
        # weight only primary physiological features (not motion)
        primary_cv = cv[:4]  # [HR, SpO2, Temp, BP]
        mean_cv = float(np.mean(primary_cv))
        return float(np.clip(1.0 - mean_cv * 4.0, 0.0, 1.0))

    def _finalise_calibration(self) -> None:
        """Lock in the baseline and create the SigmaCalculator."""
        mean = self._estimator.mean
        std  = self._estimator.std
        # Enforce minimum std floors to prevent over-sensitive alerts
        # on rock-stable features (e.g. temperature change < 0.1°C)
        MIN_STD = np.array([0.02, 0.01, 0.005, 0.015, 0.05], dtype=np.float64)
        std = np.maximum(std, MIN_STD)

        self._sigma_calc        = SigmaCalculator(mean, std)
        self.is_calibrated      = True
        self.calibration_quality = self._compute_quality()

    def summary(self) -> str:
        """Human-readable summary for display."""
        if not self.is_calibrated:
            n = self._estimator.count
            return (f"[{self.patient_id}] Calibrating: {n}/{self.window_size} "
                    f"({n/self.window_size*100:.0f}%)")
        from ml.shared.preprocessing import denormalize_feature_vector
        m = denormalize_feature_vector(self.baseline_mean.astype(np.float32))
        return (
            f"[{self.patient_id}] Calibrated (quality={self.calibration_quality:.2f}) | "
            f"BaselineHR={m[0]:.1f}BPM  SpO2={m[1]:.1f}%  Temp={m[2]:.2f}°C"
        )
