"""
Synera 2.0 — Patient Data Simulator
Generates scientifically plausible vital sign streams for all 6 patient archetypes.
Every reading mirrors the exact MQTT JSON payload structure from the firmware.

Usage:
    from scripts.data_gen.mock_simulator import PatientSimulator
    from ml.shared.constants import ARCHETYPE_TRAJECTORY_ACCEL

    sim = PatientSimulator(ARCHETYPE_TRAJECTORY_ACCEL, patient_id="PT-0031", seed=42)
    for payload in sim.generate_stream(n_readings=360):
        process(payload)
"""
from __future__ import annotations

import math
import time
import random
import sys
import os
from typing import Generator, Iterator, Tuple, Optional, List
import numpy as np

# ── path bootstrap (allows running as a script from repo root) ────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.data_gen.patient_profiles import get_profile, ArchetypeProfile
from ml.shared.constants import (
    PHYSIOLOGICAL_NOISE_COV,
    ARCHETYPE_ARTIFACT_GLITCH,
    ARCHETYPE_EXERTION,
    ARCHETYPE_TRAJECTORY_ACCEL,
    ARCHETYPE_SLOW_DRIFT,
    ARCHETYPE_STABLE,
    ARCHETYPE_POST_OP_RECOVERY,
    SAMPLING_INTERVAL_SEC,
    CALIBRATION_WINDOW_READINGS,
)
from ml.shared.preprocessing import normalize_feature_vector, normalize_bp


# ─────────────────────────────────────────────────────────────────────────────

class PatientSimulator:
    """
    Generates a physiologically realistic vital sign stream for one patient.

    The simulator models:
    - Correlated physiological noise (covariance matrix from clinical literature)
    - Sinusoidal baseline drift (circadian-scale fluctuations)
    - Motion burst events (bathroom walk, repositioning)
    - Trajectory acceleration (exponential HR rise + SpO2 fall at rest)
    - Artifact spikes from sensor displacement
    - Personal baseline variation between patients of the same archetype
    """

    def __init__(
        self,
        archetype: str,
        patient_id: str,
        seed: int = 42,
        personal_variation: float = 0.5,
    ):
        """
        Args:
            archetype:           One of the ARCHETYPE_* constants.
            patient_id:          String ID propagated into every payload (e.g. "PT-0031").
            seed:                RNG seed for full reproducibility.
            personal_variation:  0–1 scale factor for inter-patient physiological variety.
                                  0 = all patients identical; 1 = maximum individual variation.
        """
        self.archetype   = archetype
        self.patient_id  = patient_id
        self.seed        = seed
        self.profile     = get_profile(archetype)
        self._rng        = np.random.default_rng(seed)
        self._py_rng     = random.Random(seed)

        # ── Personal baseline (slight variation per patient around archetype mean) ─
        var = personal_variation
        self._baseline_hr   = self.profile.rest_hr_mean   + self._rng.normal(0, var * 4)
        self._baseline_spo2 = self.profile.rest_spo2_mean + self._rng.normal(0, var * 0.5)
        self._baseline_temp = self.profile.rest_temp_mean + self._rng.normal(0, var * 0.1)
        self._baseline_sys  = self.profile.rest_sys_bp    + self._rng.normal(0, var * 5)
        self._baseline_dia  = self.profile.rest_dia_bp    + self._rng.normal(0, var * 3)

        # ── State tracking ────────────────────────────────────────────────────
        self._reading_count       = 0
        self._calibration_complete = False
        self._battery_pct         = self._py_rng.randint(60, 99)
        # Next glitch reading index (for ARTIFACT_GLITCH archetype)
        self._next_glitch_at      = self._py_rng.randint(
            self.profile.glitch_interval_min,
            self.profile.glitch_interval_max,
        )
        # Motion burst state (for EXERTION archetype)
        self._in_burst       = False
        self._burst_end      = 0
        self._next_burst_at  = self.profile.stable_readings_before_event

    # ─────────────────────────────────────────────────────────────────────────
    # Core reading generators
    # ─────────────────────────────────────────────────────────────────────────

    def generate_reading(self, timestamp: Optional[int] = None) -> dict:
        """
        Generate one reading and return the full MQTT payload dict.
        Timestamp defaults to current Unix time if not supplied.
        """
        if timestamp is None:
            timestamp = int(time.time()) + self._reading_count * SAMPLING_INTERVAL_SEC

        n = self._reading_count
        profile = self.profile

        # ── Decide whether this reading is a glitch ────────────────────────
        is_glitch = (
            profile.hr_trajectory == "glitch_spike"
            and n == self._next_glitch_at
        )
        if is_glitch:
            self._next_glitch_at = n + self._py_rng.randint(
                profile.glitch_interval_min,
                profile.glitch_interval_max,
            )

        # ── Compute base vitals for this reading ──────────────────────────
        hr, spo2, temp, sys_bp, dia_bp, motion = self._compute_vitals(n, is_glitch)

        # ── Calibration flag ──────────────────────────────────────────────
        if n >= CALIBRATION_WINDOW_READINGS:
            self._calibration_complete = True
        # Drain battery slowly
        if n % 720 == 0 and self._battery_pct > 5:
            self._battery_pct = max(5, self._battery_pct - self._py_rng.randint(0, 2))

        payload = {
            "patient_id": self.patient_id,
            "timestamp": timestamp,
            "firmware_ver": "2.0.1",
            "vitals": {
                "heart_rate":   round(hr, 1),
                "spo2":         round(spo2, 1),
                "temperature":  round(temp, 2),
                "sys_bp_est":   round(sys_bp, 1),
                "dia_bp_est":   round(dia_bp, 1),
            },
            "context": {
                "motion_score": round(motion, 2),
                "is_active":    motion > 4.0,
                "battery_pct":  self._battery_pct,
            },
            "tinyml": {
                "reconstruction_error": None,  # filled post-inference
                "pre_alert":            None,
                "calibration_complete": self._calibration_complete,
            },
        }

        self._reading_count += 1
        return payload

    def generate_stream(
        self,
        n_readings: int,
        interval_seconds: int = SAMPLING_INTERVAL_SEC,
        start_timestamp: Optional[int] = None,
    ) -> Generator[dict, None, None]:
        """
        Yields n_readings payloads spaced interval_seconds apart.
        Non-blocking (no actual sleep).  For real-time use, caller sleeps.
        """
        t0 = start_timestamp if start_timestamp is not None else int(time.time())
        for i in range(n_readings):
            yield self.generate_reading(timestamp=t0 + i * interval_seconds)

    def generate_training_batch(
        self,
        n_patients: int = 50,
        readings_per_patient: int = 720,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of normalized sliding-window tensors for training.

        Returns:
            X      — shape (N_windows, 10, 5) float32, values in [0,1]
            labels — shape (N_windows,) int   0=normal, 1=anomalous
                     (anomaly labels not used during autoencoder training,
                      but stored for evaluation ROC/AUC calculation)
        """
        from ml.shared.preprocessing import extract_windows

        all_windows: List[np.ndarray] = []
        all_labels:  List[np.ndarray] = []

        for p_idx in range(n_patients):
            sim = PatientSimulator(
                archetype=self.archetype,
                patient_id=f"TRAIN-{p_idx:04d}",
                seed=self.seed + p_idx * 100,
                personal_variation=0.6,
            )
            raw_readings = list(sim.generate_stream(readings_per_patient))

            # Build (T, 5) array of normalized features
            feature_rows = []
            for r in raw_readings:
                v = r["vitals"]
                c = r["context"]
                feat = normalize_feature_vector(
                    v["heart_rate"], v["spo2"], v["temperature"],
                    v["sys_bp_est"], v["dia_bp_est"], c["motion_score"],
                )
                feature_rows.append(feat)
            feature_matrix = np.stack(feature_rows, axis=0)  # (T, 5)

            windows = extract_windows(feature_matrix, window_size=10, step=1)  # (W, 10, 5)

            # Label: stable archetypes → all 0; deterioration → depends on position
            labels = self._label_windows(sim, readings_per_patient, len(windows))

            all_windows.append(windows)
            all_labels.append(labels)

        X      = np.concatenate(all_windows, axis=0)
        labels = np.concatenate(all_labels,  axis=0)
        return X, labels

    # ─────────────────────────────────────────────────────────────────────────
    # Internal physiology engine
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_vitals(
        self, n: int, is_glitch: bool
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Computes raw physiological values for reading n.
        Returns: (hr, spo2, temp, sys_bp, dia_bp, motion_score)
        """
        profile  = self.profile
        rng      = self._rng

        # ── 1. Generate correlated Gaussian noise ─────────────────────────
        noise = rng.multivariate_normal(np.zeros(5), PHYSIOLOGICAL_NOISE_COV)
        hr_noise, spo2_noise, temp_noise, bp_noise, motion_noise = noise

        # ── 2. Sinusoidal baseline drift (slow circadian-scale) ───────────
        period_readings = int(120 / SAMPLING_INTERVAL_SEC)  # 2-minute period
        drift_hr = 3.0 * math.sin(2 * math.pi * n / period_readings)

        # ── 3. Compute trajectory offset ─────────────────────────────────
        traj_hr_offset   = 0.0
        traj_spo2_offset = 0.0
        traj_temp_offset = 0.0
        traj_sys_offset  = 0.0
        traj_dia_offset  = 0.0
        trajectory_motion = None  # None = use default

        # GLITCH: sensor artifact
        if is_glitch:
            hr   = profile.glitch_hr_spike  + rng.uniform(-5, 5)
            spo2 = profile.glitch_spo2_drop + rng.uniform(-3, 3)
            temp = self._baseline_temp + temp_noise * 0.3
            sys_bp = 180.0 + rng.uniform(-10, 10)
            dia_bp = 110.0 + rng.uniform(-5, 5)
            motion = float(np.clip(abs(motion_noise) * 3 + 6, 5, 10))
            return hr, spo2, temp, sys_bp, dia_bp, motion

        # EXERTION: burst of high motion + HR
        if profile.hr_trajectory == "exertion_burst":
            if n == self._next_burst_at and not self._in_burst:
                self._in_burst   = True
                self._burst_end  = n + profile.trajectory_duration_readings
            if self._in_burst:
                if n < self._burst_end:
                    progress = (n - self._next_burst_at) / max(1, profile.trajectory_duration_readings)
                    # Smooth trapezoid ramp: ramp up, plateau, ramp down
                    if progress < 0.2:
                        factor = progress / 0.2
                    elif progress > 0.8:
                        factor = (1.0 - progress) / 0.2
                    else:
                        factor = 1.0
                    hr_delta          = (profile.trajectory_target_hr - self._baseline_hr) * factor
                    traj_hr_offset    = hr_delta
                    traj_spo2_offset  = -profile.spo2_drop_total * factor
                    trajectory_motion = profile.exertion_motion_level * factor + abs(motion_noise)
                else:
                    # Recovery: return to baseline over next 60 readings
                    recovery_n = n - self._burst_end
                    if recovery_n < 60:
                        factor = max(0.0, 1.0 - recovery_n / 60.0)
                        traj_hr_offset   = (profile.trajectory_target_hr - self._baseline_hr) * factor * 0.5
                        traj_spo2_offset = -profile.spo2_drop_total * factor * 0.3
                    else:
                        self._in_burst       = False
                        # Schedule next burst: random re-burst for extended simulations
                        self._next_burst_at  = n + self._py_rng.randint(180, 360)

        # EXPONENTIAL trajectory (silent deterioration)
        elif profile.hr_trajectory == "exponential":
            event_start = profile.stable_readings_before_event
            if n >= event_start:
                elapsed   = n - event_start
                duration  = max(1, profile.trajectory_duration_readings)
                # Exponential: HR(t) = baseline + Δ * (e^(k*t/dur) - 1) / (e^k - 1)
                k         = 2.5   # controls how quickly curve steepens
                base_delta = profile.trajectory_target_hr - self._baseline_hr
                exp_factor = (math.exp(k * min(elapsed, duration) / duration) - 1) / (math.exp(k) - 1)
                traj_hr_offset   = base_delta * exp_factor
                traj_spo2_offset = -profile.spo2_drop_total * exp_factor
                traj_temp_offset =  profile.temp_rise_total * exp_factor
                traj_sys_offset  = 20.0 * exp_factor   # BP rises with HR
                traj_dia_offset  = 12.0 * exp_factor

        # LINEAR trajectory (slow drift)
        elif profile.hr_trajectory == "linear":
            duration = max(1, profile.trajectory_duration_readings)
            progress = min(n / duration, 1.0)
            delta_hr = profile.trajectory_target_hr - self._baseline_hr
            traj_hr_offset   = delta_hr * progress
            traj_spo2_offset = -profile.spo2_drop_total * progress

        # ── 4. Assemble final values ──────────────────────────────────────
        hr     = float(np.clip(
            self._baseline_hr + drift_hr + traj_hr_offset + hr_noise * 0.4,
            30, 220,
        ))
        spo2   = float(np.clip(
            self._baseline_spo2 + traj_spo2_offset + spo2_noise * 0.3,
            85, 100,
        ))
        temp   = float(np.clip(
            self._baseline_temp + traj_temp_offset + temp_noise * 0.15,
            35, 42,
        ))
        sys_bp = float(np.clip(
            self._baseline_sys + traj_sys_offset + bp_noise * 0.8,
            70, 200,
        ))
        dia_bp = float(np.clip(
            self._baseline_dia + traj_dia_offset + bp_noise * 0.5,
            40, 120,
        ))

        if trajectory_motion is not None:
            motion = float(np.clip(trajectory_motion, 0, 10))
        else:
            # Resting: low motion with occasional small bursts
            base_motion = profile.rest_motion_mean
            if self._py_rng.random() < 0.02:   # 2% chance of small shift
                base_motion += self._py_rng.uniform(1.5, 3.0)
            motion = float(np.clip(
                base_motion + abs(motion_noise) * 0.3,
                0, 10,
            ))

        return hr, spo2, temp, sys_bp, dia_bp, motion

    # ─────────────────────────────────────────────────────────────────────────
    # Label helper for training batch
    # ─────────────────────────────────────────────────────────────────────────

    def _label_windows(
        self,
        sim: "PatientSimulator",
        total_readings: int,
        n_windows: int,
    ) -> np.ndarray:
        """
        Produce anomaly labels for each sliding window.
        0 = normal, 1 = anomalous.
        Only TRAJECTORY_ACCEL archetype produces anomalous windows.
        """
        labels = np.zeros(n_windows, dtype=np.int32)
        if self.archetype == ARCHETYPE_TRAJECTORY_ACCEL:
            event_start = sim.profile.stable_readings_before_event
            for i in range(n_windows):
                window_end = i + 10  # window covers readings i..i+9
                if window_end > event_start:
                    labels[i] = 1
        return labels

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience helpers
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset simulator state to beginning. Optionally reseed."""
        new_seed = seed if seed is not None else self.seed
        self.__init__(self.archetype, self.patient_id, new_seed)

    @property
    def personal_baseline(self) -> dict:
        """Return this patient's personal baseline for display / logging."""
        return {
            "hr":          round(self._baseline_hr, 1),
            "spo2":        round(self._baseline_spo2, 1),
            "temperature": round(self._baseline_temp, 2),
            "sys_bp":      round(self._baseline_sys, 1),
            "dia_bp":      round(self._baseline_dia, 1),
        }

    def __repr__(self) -> str:
        return (
            f"PatientSimulator(archetype={self.archetype!r}, "
            f"id={self.patient_id!r}, seed={self.seed}, "
            f"readings={self._reading_count})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ml.shared.constants import ALL_ARCHETYPES

    print("Synera 2.0 — PatientSimulator smoke test\n")
    for archetype in ALL_ARCHETYPES:
        sim = PatientSimulator(archetype, patient_id="TEST-001", seed=0)
        readings = list(sim.generate_stream(20))
        hr_values = [r["vitals"]["heart_rate"] for r in readings]
        print(f"  {archetype:25s} HR[0..19]: {[round(h) for h in hr_values]}")
    print("\nAll archetypes OK.")
