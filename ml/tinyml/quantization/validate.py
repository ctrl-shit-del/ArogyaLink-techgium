"""
Synera 2.0 — Quantization Validation
Compares FP32 (PyTorch) vs INT8 (ONNX) reconstruction errors and accuracy.

Metrics printed:
  - Mean absolute error difference (FP32 vs INT8 reconstruction MSE)
  - AUC-ROC on anomaly detection using both model flavours
  - Size and inference-time comparison
  - Pass/fail table for ESP32-S3 deployment gate

No TensorFlow required — uses onnxruntime for INT8 inference.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ── repo-root bootstrap ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]   # ml/tinyml/quantization → ml/tinyml → ml → ArogyaLink
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

# ── Deployment-gate thresholds ────────────────────────────────────────────────
MAX_MSE_DELTA    = 0.005          # absolute MSE difference FP32 vs INT8
MAX_AUC_DROP     = 0.02           # INT8 AUC may not fall more than 2 % below FP32
MAX_MODEL_SIZE_KB = 50.0          # fits ESP32-S3 flash budget


def _run_pytorch_inference(
    model_path: Path,
    windows:    np.ndarray,
) -> tuple[np.ndarray, float]:
    """Returns (per_sample_mse, latency_ms_per_sample)."""
    import torch
    from ml.tinyml.model.autoencoder import VitalAutoencoder

    model = VitalAutoencoder.load(model_path)
    model.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        x      = torch.from_numpy(windows.astype(np.float32))
        # model.forward() returns (reconstructed, latent) — unpack correctly
        x_hat, _ = model(x)
        errors = torch.mean((x_hat - x) ** 2, dim=(1, 2)).numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    latency = elapsed_ms / len(windows)
    return errors, latency


def _run_onnx_inference(
    onnx_path: Path,
    windows:   np.ndarray,
) -> tuple[np.ndarray, float]:
    """Returns (per_sample_mse, latency_ms_per_sample) using onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")

    sess       = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    errors = []
    t0 = time.perf_counter()
    for i in range(len(windows)):
        sample = windows[i : i + 1].astype(np.float32)          # (1, 10, 5)
        outputs = sess.run(None, {input_name: sample})
        # outputs[0] = reconstruction (1, 10, 5), outputs[1] = latent (if exported)
        recon = outputs[0]
        mse   = float(np.mean((recon - sample) ** 2))
        errors.append(mse)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    latency = elapsed_ms / len(windows)
    return np.array(errors, dtype=np.float32), latency


def compute_auc(errors: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC-ROC using sklearn."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, errors))
    except ImportError:
        # Fallback: approximate AUC via sorted threshold sweep
        sorted_idx = np.argsort(errors)[::-1]
        tp = fp = 0
        prev_fp = prev_tp = 0
        auc = 0.0
        n_pos = int(np.sum(labels))
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        for idx in sorted_idx:
            if labels[idx] == 1:
                tp += 1
            else:
                fp += 1
                auc += (tp + prev_tp) / 2 * (1.0 / n_neg)
                prev_tp = tp
                prev_fp = fp
        return auc


def verify_quantization(
    pytorch_checkpoint: str | Path = "ml/tinyml/checkpoints/autoencoder_final.pt",
    tflite_path:        str | Path = "ml/tinyml/checkpoints/autoencoder_int8.tflite",
    n_eval_windows:     int = 500,
) -> dict:
    """
    Run full FP32 vs INT8 comparison and print formatted report.

    Returns:
        dict with all metrics and pass/fail status.
    """
    from scripts.data_gen.scenario_generator import prepare_evaluation_data

    pt_path  = Path(_ROOT) / pytorch_checkpoint
    tf_path  = Path(_ROOT) / tflite_path

    print("\n" + "=" * 60)
    print("  Synera 2.0 — Quantization Verification Report")
    print("=" * 60)

    # ── Load evaluation data ─────────────────────────────────────────────────
    print("  Generating evaluation windows …")
    eval_data = prepare_evaluation_data(n_patients_per_archetype=5)

    all_windows = []
    all_labels  = []
    for archetype, (X, y) in eval_data.items():
        # Cap to avoid OOM
        cap = min(len(X), n_eval_windows // len(eval_data))
        all_windows.append(X[:cap])
        all_labels.append(y[:cap])

    windows = np.concatenate(all_windows, axis=0)
    labels  = np.concatenate(all_labels,  axis=0)
    print(f"  Eval windows : {len(windows):,}  (normal={int(np.sum(labels==0))}, anomaly={int(np.sum(labels==1))})")

    # ── Run inference ────────────────────────────────────────────────────────
    print("\n  Running FP32 inference (PyTorch) …")
    fp32_errors, fp32_lat = _run_pytorch_inference(pt_path, windows)
    fp32_auc = compute_auc(fp32_errors, labels)

    print("  Running INT8 inference (ONNX Runtime) …")
    int8_errors, int8_lat = _run_onnx_inference(tf_path, windows)
    int8_auc = compute_auc(int8_errors, labels)

    # ── Metrics ──────────────────────────────────────────────────────────────
    mse_delta  = float(np.mean(np.abs(fp32_errors - int8_errors)))
    auc_drop   = fp32_auc - int8_auc
    model_kb   = tf_path.stat().st_size / 1024 if tf_path.exists() else float("inf")

    # ── Report ───────────────────────────────────────────────────────────────
    PASS = "PASS ✓"
    FAIL = "FAIL ✗"
    rows = [
        ("FP32 AUC-ROC",       f"{fp32_auc:.4f}",   "—",                  PASS),
        ("INT8 AUC-ROC",       f"{int8_auc:.4f}",   "—",                  PASS),
        ("AUC Drop",           f"{auc_drop:.4f}",   f"< {MAX_AUC_DROP}",  PASS if auc_drop < MAX_AUC_DROP else FAIL),
        ("Mean |ΔMSE|",        f"{mse_delta:.5f}",  f"< {MAX_MSE_DELTA}", PASS if mse_delta < MAX_MSE_DELTA else FAIL),
        ("INT8 ONNX size",     f"{model_kb:.1f} KB",f"< {MAX_MODEL_SIZE_KB} KB", PASS if model_kb < MAX_MODEL_SIZE_KB else FAIL),
        ("FP32 latency/sample",f"{fp32_lat:.2f} ms","—",                  PASS),
        ("INT8 latency/sample",f"{int8_lat:.2f} ms","—",                  PASS),
    ]

    col_w = [28, 14, 16, 10]
    divider = "  " + "-" * (sum(col_w) + 9)
    header  = "  {:<{}} {:<{}} {:<{}} {:<{}}".format(
        "Metric", col_w[0], "Value", col_w[1], "Threshold", col_w[2], "Status", col_w[3]
    )
    print(f"\n{header}")
    print(divider)
    for name, value, thresh, status in rows:
        print("  {:<{}} {:<{}} {:<{}} {:<{}}".format(
            name, col_w[0], value, col_w[1], thresh, col_w[2], status, col_w[3]
        ))
    print(divider)

    all_pass = all(r[3] == PASS for r in rows if r[2] != "—")
    verdict  = "ALL CHECKS PASSED — Model ready for ESP32-S3 deployment" if all_pass else "DEPLOYMENT GATE FAILED — Review metrics above"
    print(f"\n  {verdict}\n")

    return {
        "fp32_auc":    fp32_auc,
        "int8_auc":    int8_auc,
        "auc_drop":    auc_drop,
        "mse_delta":   mse_delta,
        "model_kb":    model_kb,
        "fp32_lat_ms": fp32_lat,
        "int8_lat_ms": int8_lat,
        "all_pass":    all_pass,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify INT8 quantization quality")
    parser.add_argument("--checkpoint",   default="ml/tinyml/checkpoints/autoencoder_final.pt")
    parser.add_argument("--tflite",       default="ml/tinyml/checkpoints/autoencoder_int8.tflite")
    parser.add_argument("--n-eval",       type=int, default=500)
    args = parser.parse_args()

    results = verify_quantization(args.checkpoint, args.tflite, args.n_eval)
    sys.exit(0 if results["all_pass"] else 1)
