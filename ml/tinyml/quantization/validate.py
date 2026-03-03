"""Validate ONNX autoencoder quantization quality.

Computes ROC-AUC, precision/recall, and MSE stats for both the fp32 and
int8 ONNX models.  Generates normal windows (baseline noise) and anomaly
windows (trajectory-acceleration pattern) without needing a live DB.

Run:
    python -m ml.tinyml.quantization.validate
"""
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# --- Paths -----------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[4]          # repo root
_MODELS_DIR = _ROOT / "models"
_INT8_PATH = _MODELS_DIR / "autoencoder_int8.onnx"
_FP32_PATH  = _MODELS_DIR / "autoencoder.onnx"
_CONFIG_PATH = _MODELS_DIR / "tinyml_config.json"

_WINDOW_LEN = 10
_N_FEATURES  = 4   # heart_rate, spo2, temperature, motion_score

_FEATURE_MIN = np.array([40.0,  85.0, 35.0, 0.0],  dtype=np.float32)
_FEATURE_MAX = np.array([200.0, 100.0, 41.0, 10.0], dtype=np.float32)


def _normalise(x: np.ndarray) -> np.ndarray:
    denom = _FEATURE_MAX - _FEATURE_MIN
    denom[denom == 0] = 1.0
    return (x - _FEATURE_MIN) / denom


# --- Synthetic data generators ----------------------------------------

def _normal_window(baseline_hr=80.0, baseline_spo2=97.0, baseline_temp=37.0) -> np.ndarray:
    """Stable patient — Gaussian noise around baseline."""
    matrix = np.zeros((_WINDOW_LEN, _N_FEATURES), dtype=np.float32)
    for i in range(_WINDOW_LEN):
        matrix[i, 0] = baseline_hr   + random.gauss(0, 3.0)
        matrix[i, 1] = np.clip(baseline_spo2 + random.gauss(0, 0.8), 85, 100)
        matrix[i, 2] = baseline_temp + random.gauss(0, 0.15)
        matrix[i, 3] = random.randint(0, 2)
    return matrix


def _anomaly_window(baseline_hr=80.0, baseline_spo2=98.0, baseline_temp=37.0) -> np.ndarray:
    """Trajectory-acceleration pattern — HR rises with increasing rate (Rule C scenario)."""
    matrix = np.zeros((_WINDOW_LEN, _N_FEATURES), dtype=np.float32)
    end_hr    = baseline_hr + random.uniform(30, 55)
    end_spo2  = baseline_spo2 - random.uniform(6, 12)
    end_temp  = baseline_temp + random.uniform(0.8, 1.4)
    for i in range(_WINDOW_LEN):
        t = (i + 1) / _WINDOW_LEN
        # Exponential acceleration: faster rise toward the end
        x = (math.exp(t * 2) - 1) / (math.e ** 2 - 1)
        matrix[i, 0] = baseline_hr   + (end_hr   - baseline_hr)   * x + random.gauss(0, 1.5)
        matrix[i, 1] = np.clip(baseline_spo2 + (end_spo2 - baseline_spo2) * x, 85, 100)
        matrix[i, 2] = baseline_temp + (end_temp - baseline_temp) * x
        matrix[i, 3] = random.randint(0, 1)   # low motion — physiological
    return matrix


def _build_dataset(n_normal: int = 300, n_anomaly: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Return (windows, labels) where 0=normal, 1=anomaly."""
    normal_windows  = [_normalise(_normal_window())  for _ in range(n_normal)]
    anomaly_windows = [_normalise(_anomaly_window()) for _ in range(n_anomaly)]
    windows = np.stack(normal_windows + anomaly_windows, axis=0)  # (N, 10, 4)
    labels  = np.array([0] * n_normal + [1] * n_anomaly, dtype=np.int32)
    return windows, labels


# --- ONNX inference ---------------------------------------------------

def _load_session(model_path: Path):
    try:
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        return ort.InferenceSession(str(model_path), sess_opts)
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)


def _reconstruction_errors(session, windows: np.ndarray) -> np.ndarray:
    """Batch inference; returns per-sample MSE array."""
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    errors = []
    for i in range(len(windows)):
        x = windows[i : i + 1].astype(np.float32)          # (1, 10, 4)
        recon = session.run([output_name], {input_name: x})[0]
        mse   = float(np.mean((x - recon) ** 2))
        errors.append(mse)
    return np.array(errors, dtype=np.float64)


# --- Metrics ----------------------------------------------------------

def _compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC — no sklearn dependency."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_pos, n_neg = len(pos), len(neg)
    # Count concordant pairs
    concordant = sum(
        1 + (0.5 if p == n else 0)
        for p in pos
        for n in neg
        if p > n
    )
    # Vectorised version for speed
    concordant = float(np.sum(pos[:, None] > neg[None, :]))
    concordant += 0.5 * float(np.sum(pos[:, None] == neg[None, :]))
    return concordant / (n_pos * n_neg)


def _precision_recall_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> Tuple[float, float, float]:
    preds = (scores > threshold).astype(np.int32)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _validate_model(model_path: Path, threshold: float, windows: np.ndarray, labels: np.ndarray, tag: str):
    if not model_path.exists():
        print(f"  [{tag}] SKIP — model not found: {model_path}")
        return

    print(f"\n  [{tag}] {model_path.name}")
    session = _load_session(model_path)
    errors  = _reconstruction_errors(session, windows)

    normal_errors  = errors[labels == 0]
    anomaly_errors = errors[labels == 1]

    print(f"    Normal  MSE: mean={normal_errors.mean():.5f}  std={normal_errors.std():.5f}  max={normal_errors.max():.5f}")
    print(f"    Anomaly MSE: mean={anomaly_errors.mean():.5f}  std={anomaly_errors.std():.5f}  min={anomaly_errors.min():.5f}")

    auc = _compute_roc_auc(labels, errors)
    print(f"    ROC-AUC: {auc:.4f}  {'PASS ✓' if auc >= 0.80 else 'FAIL ✗'}")

    precision, recall, f1 = _precision_recall_at_threshold(labels, errors, threshold)
    print(f"    @ threshold={threshold:.5f}  precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")

    # Suggest optimal threshold (Youden's J)
    best_j, best_thresh = -1.0, threshold
    unique_thresholds = np.percentile(errors, np.arange(0, 101, 2))
    for t in unique_thresholds:
        p, r, _ = _precision_recall_at_threshold(labels, errors, t)
        tpr = r
        tnr = 1.0 - (np.sum((errors > t) & (labels == 0)) / max(1, np.sum(labels == 0)))
        j = tpr + tnr - 1
        if j > best_j:
            best_j, best_thresh = j, t
    print(f"    Optimal threshold (Youden J={best_j:.3f}): {best_thresh:.5f}")


def main():
    print("=" * 60)
    print("  Synera TinyML — ONNX quantization validation")
    print("=" * 60)

    # Load threshold from config
    threshold = 0.00168
    if _CONFIG_PATH.exists():
        cfg = json.loads(_CONFIG_PATH.read_text())
        threshold = float(cfg.get("threshold", threshold))
        print(f"\n  Config: threshold={threshold}  val_loss={cfg.get('val_loss')}  params={cfg.get('params')}")

    print(f"\n  Generating dataset: 300 normal + 300 anomaly windows...")
    random.seed(42)
    np.random.seed(42)
    windows, labels = _build_dataset(n_normal=300, n_anomaly=300)
    print(f"  Dataset shape: {windows.shape}  Labels: {dict(zip(*np.unique(labels, return_counts=True)))}")

    _validate_model(_FP32_PATH,  threshold, windows, labels, "FP32")
    _validate_model(_INT8_PATH,  threshold, windows, labels, "INT8")

    print("\n" + "=" * 60)
    print("  Validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
