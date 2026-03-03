"""
Synera 2.0 — Preprocessing / Normalization
All normalization is min-max to [0, 1] using physiological bounds from constants.
These functions are mirrored exactly in the ESP32-S3 firmware (C equivalents).
"""
from __future__ import annotations

import numpy as np
from ml.shared.constants import FEATURE_RANGES, SYS_BP_MIN, SYS_BP_MAX, DIA_BP_MIN, DIA_BP_MAX


# ─────────────────────────────────────────────────────────────────────────────
# Scalar normalizers
# ─────────────────────────────────────────────────────────────────────────────

def normalize(value: float, min_val: float, max_val: float) -> float:
    """Min-max normalize a scalar to [0, 1]. Clamps to [0, 1] on out-of-range input."""
    if max_val <= min_val:
        raise ValueError(f"max_val ({max_val}) must be > min_val ({min_val})")
    normalized = (value - min_val) / (max_val - min_val)
    return float(np.clip(normalized, 0.0, 1.0))


def denormalize(value: float, min_val: float, max_val: float) -> float:
    """Inverse of normalize — recovers raw physiological value."""
    return float(np.clip(value, 0.0, 1.0)) * (max_val - min_val) + min_val


def normalize_heart_rate(bpm: float) -> float:
    r = FEATURE_RANGES["heart_rate"]
    return normalize(bpm, r["min"], r["max"])


def normalize_spo2(pct: float) -> float:
    r = FEATURE_RANGES["spo2"]
    return normalize(pct, r["min"], r["max"])


def normalize_temperature(celsius: float) -> float:
    r = FEATURE_RANGES["temperature"]
    return normalize(celsius, r["min"], r["max"])


def normalize_bp(sys_bp: float, dia_bp: float) -> float:
    """
    Composite blood pressure score — combines systolic and diastolic into a
    single normalized [0,1] feature.  Formula: (sys/200 + dia/120) / 2
    This matches the firmware implementation exactly.
    """
    sys_norm = normalize(sys_bp, SYS_BP_MIN, SYS_BP_MAX)
    dia_norm = normalize(dia_bp, DIA_BP_MIN, DIA_BP_MAX)
    return (sys_norm + dia_norm) / 2.0


def normalize_motion(score: float) -> float:
    r = FEATURE_RANGES["motion_score"]
    return normalize(score, r["min"], r["max"])


# ─────────────────────────────────────────────────────────────────────────────
# Full feature vector normalizer
# ─────────────────────────────────────────────────────────────────────────────

def normalize_feature_vector(
    heart_rate: float,
    spo2: float,
    temperature: float,
    sys_bp: float,
    dia_bp: float,
    motion_score: float,
) -> np.ndarray:
    """
    Converts raw sensor values → normalized 5-d feature vector.
    Output order: [hr_norm, spo2_norm, temp_norm, bp_norm, motion_norm]
    Shape: (5,)
    """
    return np.array([
        normalize_heart_rate(heart_rate),
        normalize_spo2(spo2),
        normalize_temperature(temperature),
        normalize_bp(sys_bp, dia_bp),
        normalize_motion(motion_score),
    ], dtype=np.float32)


def normalize_window(window: np.ndarray) -> np.ndarray:
    """
    Normalize a (10, 5) raw window.
    Applies per-feature normalization using physiological bounds.
    Returns (10, 5) float32 array normalized to [0, 1].
    NOTE: The 5th column (estimated_bp) is assumed to already be
          the composite score (pre-computed by normalize_bp).
    """
    assert window.ndim == 2 and window.shape[1] == 5, f"Expected (N, 5), got {window.shape}"
    bounds = [
        (FEATURE_RANGES["heart_rate"]["min"],   FEATURE_RANGES["heart_rate"]["max"]),
        (FEATURE_RANGES["spo2"]["min"],          FEATURE_RANGES["spo2"]["max"]),
        (FEATURE_RANGES["temperature"]["min"],   FEATURE_RANGES["temperature"]["max"]),
        (0.0, 1.0),   # bp composite is already [0,1]
        (FEATURE_RANGES["motion_score"]["min"],  FEATURE_RANGES["motion_score"]["max"]),
    ]
    out = np.empty_like(window, dtype=np.float32)
    for i, (mn, mx) in enumerate(bounds):
        out[:, i] = np.clip((window[:, i] - mn) / (mx - mn), 0.0, 1.0)
    return out


def denormalize_feature_vector(vec: np.ndarray) -> np.ndarray:
    """
    Inverse of normalize_feature_vector.
    Input: (5,) normalized vector
    Output: (5,) approximate raw values [HR BPM, SpO2 %, Temp °C, BP score, Motion]
    """
    assert vec.shape == (5,)
    bounds = [
        (FEATURE_RANGES["heart_rate"]["min"],   FEATURE_RANGES["heart_rate"]["max"]),
        (FEATURE_RANGES["spo2"]["min"],          FEATURE_RANGES["spo2"]["max"]),
        (FEATURE_RANGES["temperature"]["min"],   FEATURE_RANGES["temperature"]["max"]),
        (0.0, 1.0),
        (FEATURE_RANGES["motion_score"]["min"],  FEATURE_RANGES["motion_score"]["max"]),
    ]
    out = np.empty(5, dtype=np.float32)
    for i, (mn, mx) in enumerate(bounds):
        out[i] = float(np.clip(vec[i], 0.0, 1.0)) * (mx - mn) + mn
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sliding window extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_windows(
    readings: np.ndarray,
    window_size: int = 10,
    step: int = 1,
) -> np.ndarray:
    """
    Convert a (T, 5) time-series of normalized readings into
    (N_windows, window_size, 5) sliding windows.

    Args:
        readings:    (T, 5) float32 array — normalized feature vectors in time order
        window_size: number of readings per window (default 10 = 50 seconds)
        step:        stride between windows (default 1 for maximum data utilisation)

    Returns:
        (N_windows, window_size, 5) float32 array
    """
    if readings.ndim != 2 or readings.shape[1] != 5:
        raise ValueError(f"Expected (T, 5), got {readings.shape}")
    T = readings.shape[0]
    if T < window_size:
        raise ValueError(f"Not enough readings ({T}) for window_size={window_size}")

    indices = range(0, T - window_size + 1, step)
    windows = np.stack([readings[i : i + window_size] for i in indices], axis=0)
    return windows.astype(np.float32)
