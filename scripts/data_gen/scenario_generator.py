"""
Synera 2.0 — Scenario Generator
Batch training data factory.  Produces (X, labels) tensors for the
autoencoder training pipeline using the PatientSimulator.
"""
from __future__ import annotations

import sys
import os
import numpy as np
from typing import Tuple, Dict, List

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.data_gen.mock_simulator import PatientSimulator
from ml.shared.constants import (
    ARCHETYPE_STABLE, ARCHETYPE_POST_OP_RECOVERY,
    ARCHETYPE_TRAJECTORY_ACCEL, ARCHETYPE_EXERTION,
    ARCHETYPE_SLOW_DRIFT, ARCHETYPE_ARTIFACT_GLITCH,
    SAMPLING_INTERVAL_SEC,
)
from ml.shared.preprocessing import extract_windows, normalize_feature_vector


def _sim_to_windows(
    archetype: str,
    n_patients: int,
    readings_per_patient: int,
    base_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Internal helper: simulate n_patients of one archetype → (windows, labels)."""
    all_w: List[np.ndarray] = []
    all_l: List[np.ndarray] = []

    for p in range(n_patients):
        sim = PatientSimulator(archetype, patient_id=f"P{p:04d}", seed=base_seed + p * 17)
        readings = list(sim.generate_stream(readings_per_patient))

        feats = np.stack([
            normalize_feature_vector(
                r["vitals"]["heart_rate"], r["vitals"]["spo2"],
                r["vitals"]["temperature"], r["vitals"]["sys_bp_est"],
                r["vitals"]["dia_bp_est"], r["context"]["motion_score"],
            )
            for r in readings
        ], axis=0)  # (T, 5)

        windows = extract_windows(feats, window_size=10, step=1)  # (W, 10, 5)

        # Labelling
        if archetype == ARCHETYPE_TRAJECTORY_ACCEL:
            event_start = sim.profile.stable_readings_before_event
            labels = np.array([
                1 if (i + 10) > event_start else 0
                for i in range(len(windows))
            ], dtype=np.int32)
        else:
            labels = np.zeros(len(windows), dtype=np.int32)

        all_w.append(windows)
        all_l.append(labels)

    return np.concatenate(all_w, axis=0), np.concatenate(all_l, axis=0)


def prepare_training_data(
    n_patients: int = 50,
    minutes_per_patient: int = 60,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate TRAINING data — ONLY normal archetypes.
    The autoencoder must never see anomalous patterns during training.

    Returns:
        X      (N_windows, 10, 5) float32 — all normal windows
        labels (N_windows,)       int32   — all zeros (all normal)
    """
    readings_per = int(minutes_per_patient * 60 / SAMPLING_INTERVAL_SEC)
    half = n_patients // 2

    w_stable, l_stable   = _sim_to_windows(ARCHETYPE_STABLE,          half,           readings_per, seed)
    w_postop, l_postop   = _sim_to_windows(ARCHETYPE_POST_OP_RECOVERY, n_patients - half, readings_per, seed + 9999)

    X      = np.concatenate([w_stable, w_postop], axis=0)
    labels = np.concatenate([l_stable, l_postop], axis=0)

    # Shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], labels[idx]


def prepare_evaluation_data(
    n_patients_per_archetype: int = 10,
    minutes_per_patient: int = 30,
    seed: int = 2024,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate EVALUATION data for all archetypes.
    Returns a dict mapping archetype name → (X, labels).
    Used for ROC-AUC, sensitivity, and false positive rate computation.
    """
    readings_per = int(minutes_per_patient * 60 / SAMPLING_INTERVAL_SEC)
    eval_archetypes = [
        ARCHETYPE_STABLE,
        ARCHETYPE_TRAJECTORY_ACCEL,
        ARCHETYPE_EXERTION,
        ARCHETYPE_SLOW_DRIFT,
        ARCHETYPE_ARTIFACT_GLITCH,
        ARCHETYPE_POST_OP_RECOVERY,
    ]
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for i, archetype in enumerate(eval_archetypes):
        w, l = _sim_to_windows(archetype, n_patients_per_archetype, readings_per, seed + i * 1000)
        result[archetype] = (w, l)
    return result


def prepare_calibration_data(
    n_windows: int = 200,
    seed: int = 777,
) -> np.ndarray:
    """
    Generate a small representative set of normal windows.
    Used as the TFLite INT8 quantization calibration dataset.
    Returns (n_windows, 10, 5) float32.
    """
    sim = PatientSimulator(ARCHETYPE_STABLE, patient_id="CAL-001", seed=seed)
    readings = list(sim.generate_stream(n_windows + 15))
    feats = np.stack([
        normalize_feature_vector(
            r["vitals"]["heart_rate"], r["vitals"]["spo2"],
            r["vitals"]["temperature"], r["vitals"]["sys_bp_est"],
            r["vitals"]["dia_bp_est"], r["context"]["motion_score"],
        )
        for r in readings
    ], axis=0)
    return extract_windows(feats, window_size=10, step=1)[:n_windows]


if __name__ == "__main__":
    print("Generating training data (50 patients × 60 min)…")
    X, labels = prepare_training_data(n_patients=50, minutes_per_patient=60)
    print(f"  Training set: X={X.shape}, labels={labels.shape}, anomaly={labels.sum()}/total={len(labels)}")

    print("Generating evaluation data…")
    eval_data = prepare_evaluation_data(n_patients_per_archetype=10, minutes_per_patient=30)
    for arch, (Xe, le) in eval_data.items():
        print(f"  {arch:25s} windows={len(Xe):6d}  anomalous={le.sum():5d}")
