"""TinyML anomaly detection service — ONNX autoencoder inference.

Loads models/autoencoder_int8.onnx (falls back to autoencoder.onnx).
Input:  window of 10 readings × 4 features [heart_rate, spo2, temperature, motion_score]
Output: reconstruction_error (float), is_anomaly (bool)

Usage:
    from backend.services.tinyml_service import tinyml_service
    error, is_anomaly = tinyml_service.score(window_readings)
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_CONFIG_PATH = _MODELS_DIR / "tinyml_config.json"
_INT8_PATH = _MODELS_DIR / "autoencoder_int8.onnx"
_FP32_PATH = _MODELS_DIR / "autoencoder.onnx"

# Feature order expected by the model (must match ml/shared/constants.py FEATURE_NAMES)
# [heart_rate, spo2, temperature, estimated_bp, motion_score]  — 5 features, NOT 4
_FEATURES = ["heart_rate", "spo2", "temperature", "estimated_bp", "motion_score"]
_WINDOW_LEN = 10
_N_FEATURES = 5

# Per-feature normalisation bounds — must exactly match ml/shared/constants.py FEATURE_RANGES
#   estimated_bp is a [0, 1] composite so its bounds are (0.0, 1.0) → normalise is a no-op
_FEATURE_MIN = np.array([30.0,  85.0, 35.0, 0.0, 0.0],  dtype=np.float32)
_FEATURE_MAX = np.array([220.0, 100.0, 42.0, 1.0, 10.0], dtype=np.float32)

# Systolic / diastolic BP bounds for the composite score
# Formula (from ml/shared/preprocessing.py): (sys_norm + dia_norm) / 2
_SYS_BP_MIN, _SYS_BP_MAX = 70.0, 200.0
_DIA_BP_MIN, _DIA_BP_MAX = 40.0, 120.0


def _normalise(x: np.ndarray) -> np.ndarray:
    """Min-max normalise a (window_len, n_features) array to [0, 1]."""
    denom = _FEATURE_MAX - _FEATURE_MIN
    denom[denom == 0] = 1.0
    return np.clip((x - _FEATURE_MIN) / denom, 0.0, 1.0)


def _bp_composite(sys_bp: float, dia_bp: float) -> float:
    """Combine sys + dia into the single normalised BP feature the model was trained on."""
    sys_norm = float(np.clip((sys_bp - _SYS_BP_MIN) / (_SYS_BP_MAX - _SYS_BP_MIN), 0.0, 1.0))
    dia_norm = float(np.clip((dia_bp - _DIA_BP_MIN) / (_DIA_BP_MAX - _DIA_BP_MIN), 0.0, 1.0))
    return (sys_norm + dia_norm) / 2.0


class TinyMLService:
    """Singleton ONNX inference wrapper."""

    def __init__(self):
        self._session = None
        self._threshold: float = 0.00168
        self._ready: bool = False
        self._load()

    def _load(self):
        """Load ONNX session and config. Logs but never raises — graceful degradation."""
        # Load config
        if _CONFIG_PATH.exists():
            try:
                cfg = json.loads(_CONFIG_PATH.read_text())
                self._threshold = float(cfg.get("threshold", self._threshold))
                logger.info(
                    "[TinyML] Config loaded — threshold=%.5f val_loss=%.5f params=%d",
                    self._threshold,
                    cfg.get("val_loss", 0),
                    cfg.get("params", 0),
                )
            except Exception as e:
                logger.warning("[TinyML] Could not read config: %s", e)

        # Try int8 first, fall back to fp32
        model_path = _INT8_PATH if _INT8_PATH.exists() else _FP32_PATH
        if not model_path.exists():
            logger.warning("[TinyML] No ONNX model found at %s — service disabled", _MODELS_DIR)
            return

        try:
            import onnxruntime as ort  # type: ignore
            # Prefer CPU — matches ESP32 semantics; no GPU needed
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(str(model_path), sess_opts)
            self._ready = True
            logger.info(
                "[TinyML] Service ready — model=%s threshold=%.5f",
                model_path.name,
                self._threshold,
            )
        except ImportError:
            logger.warning("[TinyML] onnxruntime not installed — service disabled. pip install onnxruntime")
        except Exception as e:
            logger.warning("[TinyML] Failed to load ONNX model: %s — service disabled", e)

    @property
    def ready(self) -> bool:
        return self._ready

    def score(self, readings: List[dict]) -> Tuple[float, bool]:
        """Run inference on a list of vital reading dicts.

        Args:
            readings: list of dicts with keys heart_rate, spo2, temperature, motion_score.
                      Will be padded/truncated to _WINDOW_LEN.

        Returns:
            (reconstruction_error, is_anomaly)
            Returns (0.0, False) if service is not ready.
        """
        if not self._ready or self._session is None:
            return 0.0, False

        try:
            x = self._prepare_input(readings)
            input_name = self._session.get_inputs()[0].name
            output_name = self._session.get_outputs()[0].name
            reconstructed = self._session.run([output_name], {input_name: x})[0]
            # MSE over the entire window
            error = float(np.mean((x - reconstructed) ** 2))
            is_anomaly = error > self._threshold
            return error, is_anomaly
        except Exception as e:
            logger.debug("[TinyML] Inference error: %s", e)
            return 0.0, False

    def _prepare_input(self, readings: List[dict]) -> np.ndarray:
        """Convert reading dicts → normalised (1, window_len, 5) float32 array.

        Feature order (must match ml/shared/constants.py):
          col 0: heart_rate
          col 1: spo2
          col 2: temperature
          col 3: estimated_bp  ← composite (sys_bp + dia_bp normalised, averaged)
          col 4: motion_score
        """
        # Fill with physiologically neutral defaults for missing/padding rows
        defaults = [80.0, 97.0, 37.0, _bp_composite(120.0, 80.0), 0.0]
        matrix = np.tile(defaults, (_WINDOW_LEN, 1)).astype(np.float32)

        # Overwrite with actual readings (latest _WINDOW_LEN readings, oldest first)
        n = min(len(readings), _WINDOW_LEN)
        for i, r in enumerate(readings[-n:]):
            row_idx = _WINDOW_LEN - n + i
            hr  = float(r.get("heart_rate")  or 80.0)
            sp  = float(r.get("spo2")        or 97.0)
            tmp = float(r.get("temperature") or 37.0)
            sys_bp = float(r.get("sys_bp_est") or 120.0)
            dia_bp = float(r.get("dia_bp_est") or 80.0)
            mot = float(r.get("motion_score") or 0.0)
            matrix[row_idx, 0] = hr
            matrix[row_idx, 1] = sp
            matrix[row_idx, 2] = tmp
            matrix[row_idx, 3] = _bp_composite(sys_bp, dia_bp)  # already in [0,1]
            matrix[row_idx, 4] = mot

        # Normalise cols 0,1,2,4 to [0,1]; col 3 is already [0,1] (no-op)
        matrix = _normalise(matrix)
        return matrix.reshape(1, _WINDOW_LEN, _N_FEATURES)

    def score_from_buffer(self, buffer_deque) -> Tuple[float, bool]:
        """Convenience wrapper for engine.py — accepts the in_memory_store deque."""
        return self.score(list(buffer_deque))


# Module-level singleton — imported by engine.py
tinyml_service = TinyMLService()
