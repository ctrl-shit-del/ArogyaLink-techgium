"""
Synera 2.0 — Feature Engineering
Assembles the 5-feature input vector from an MQTT payload dict
and provides feature-level utility functions.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, List

from ml.shared.preprocessing import normalize_feature_vector, normalize_bp
from ml.shared.constants import FEATURE_NAMES, N_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Primary assembly function
# ─────────────────────────────────────────────────────────────────────────────

def payload_to_feature_vector(payload: Dict[str, Any]) -> np.ndarray:
    """
    Converts a full MQTT JSON payload (as received from the wearable) into a
    normalized 5-d feature vector ready for model input.

    Expected payload structure:
        {
          "vitals": {
              "heart_rate": float,
              "spo2": float,
              "temperature": float,
              "sys_bp_est": float,
              "dia_bp_est": float,
          },
          "context": {
              "motion_score": float,
          }
        }

    Returns:
        np.ndarray shape (5,) float32, normalized to [0, 1]
        Order: [hr, spo2, temp, bp_composite, motion]
    """
    vitals  = payload["vitals"]
    context = payload["context"]
    return normalize_feature_vector(
        heart_rate=float(vitals["heart_rate"]),
        spo2=float(vitals["spo2"]),
        temperature=float(vitals["temperature"]),
        sys_bp=float(vitals["sys_bp_est"]),
        dia_bp=float(vitals["dia_bp_est"]),
        motion_score=float(context["motion_score"]),
    )


def feature_vector_to_dict(vec: np.ndarray) -> Dict[str, float]:
    """
    Maps a (5,) normalized feature vector back to a named dict.
    Useful for debugging and logging.
    """
    assert vec.shape == (N_FEATURES,), f"Expected ({N_FEATURES},), got {vec.shape}"
    return {name: float(vec[i]) for i, name in enumerate(FEATURE_NAMES)}


# ─────────────────────────────────────────────────────────────────────────────
# Stream-to-window helper
# ─────────────────────────────────────────────────────────────────────────────

def payloads_to_window(payloads: List[Dict[str, Any]]) -> np.ndarray:
    """
    Converts a list of exactly 10 consecutive MQTT payloads into a model
    input tensor of shape (1, 10, 5) ready for autoencoder inference.

    Raises:
        ValueError if len(payloads) != 10
    """
    if len(payloads) != 10:
        raise ValueError(f"Expected 10 payloads for window, got {len(payloads)}")
    vectors = np.stack([payload_to_feature_vector(p) for p in payloads], axis=0)
    return vectors[np.newaxis, :, :]   # (1, 10, 5)


# ─────────────────────────────────────────────────────────────────────────────
# BP composite helper (exposed for external use)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bp_score(sys_bp: float, dia_bp: float) -> float:
    """Composite BP score in [0, 1]. Exposed for use in simulator."""
    return normalize_bp(sys_bp, dia_bp)


# ─────────────────────────────────────────────────────────────────────────────
# Delta check for Rule A (artifact rejection)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reading_delta(
    current: Dict[str, Any],
    previous: Dict[str, Any],
) -> Dict[str, float]:
    """
    Computes absolute deltas on the three Rule A–checkable vitals between
    consecutive raw payloads.

    Returns:
        {"heart_rate": Δ, "spo2": Δ, "temperature": Δ}
    """
    def v(p: Dict) -> Dict:
        return p["vitals"]

    return {
        "heart_rate":  abs(float(v(current)["heart_rate"])  - float(v(previous)["heart_rate"])),
        "spo2":        abs(float(v(current)["spo2"])         - float(v(previous)["spo2"])),
        "temperature": abs(float(v(current)["temperature"])  - float(v(previous)["temperature"])),
    }
