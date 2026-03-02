"""
Synera 2.0 — Model Quantization Pipeline
PyTorch → ONNX (FP32) → ONNX (INT8) + Weight C-header

Conversion chain:
    VitalAutoencoder (PyTorch FP32)
        ↓  torch.onnx.export
    autoencoder.onnx              (FP32, ~29 KB, for host validation)
        ↓  onnxruntime dynamic quantization
    autoencoder_int8.onnx         (~8 KB, for ONNX Runtime inference)
        ↓  weight extraction (generate_header.py)
    autoencoder_model.h           (C float32 weight arrays for firmware)

No TensorFlow required.  Uses only torch + onnx + onnxruntime.

Dependencies:
    pip install onnx onnxruntime
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── repo-root bootstrap ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]   # ml/tinyml/quantization → ml/tinyml → ml → ArogyaLink
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ─────────────────────────────────────────────────────────────────────────────


class ModelQuantizer:
    """
    End-to-end conversion from a trained PyTorch VitalAutoencoder to
    an INT8 ONNX model + C weight-array header ready for ESP32-S3 firmware.

    Pipeline (no TensorFlow required):
        Step 1 — PyTorch ──→ FP32 ONNX      (torch.onnx.export)
        Step 2 — FP32 ONNX → INT8 ONNX      (onnxruntime dynamic quantization)
        Step 3 — checkpoint → C header       (extract float32 weights as C arrays)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        output_dir:      str | Path = "ml/tinyml/checkpoints",
    ):
        self.checkpoint_path = Path(_ROOT) / checkpoint_path
        self.output_dir      = Path(_ROOT) / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.onnx_fp32_path = self.output_dir / "autoencoder.onnx"
        self.onnx_int8_path = self.output_dir / "autoencoder_int8.onnx"
        # Keep .tflite alias so that downstream callers (validate.py, convert_to_tflite.py)
        # that expect a .tflite path still work — the file contains INT8 ONNX bytes.
        self.tflite_path    = self.onnx_int8_path

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — PyTorch → ONNX (FP32)
    # ─────────────────────────────────────────────────────────────────────────

    def pytorch_to_onnx(self) -> Path:
        """Export VitalAutoencoder to FP32 ONNX (opset 18)."""
        import torch
        from ml.tinyml.model.autoencoder import VitalAutoencoder

        print("  [1/3] Exporting PyTorch → ONNX …")
        model = VitalAutoencoder.load(self.checkpoint_path)
        model.eval()

        dummy_input = torch.zeros(1, 10, 5, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy_input,
            str(self.onnx_fp32_path),
            export_params       = True,
            opset_version       = 18,
            do_constant_folding = True,
            input_names         = ["vital_window"],
            output_names        = ["reconstruction"],
            dynamic_axes        = {"vital_window": {0: "batch"}, "reconstruction": {0: "batch"}},
        )
        size_kb = self.onnx_fp32_path.stat().st_size / 1024
        print(f"       FP32 ONNX → {self.onnx_fp32_path}  ({size_kb:.1f} KB)")
        return self.onnx_fp32_path

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — ONNX INT8 dynamic quantization via onnxruntime
    # ─────────────────────────────────────────────────────────────────────────

    def onnx_to_int8(self) -> Path:
        """Apply dynamic INT8 quantization to the ONNX model (no calibration data needed)."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            from onnxruntime.quantization.shape_inference import quant_pre_process
        except ImportError:
            raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")

        print("  [2/3] Quantising ONNX → INT8 (dynamic) …")

        # Pre-process model: add shape information required by quantizer
        preprocessed_path = self.output_dir / "autoencoder_prep.onnx"
        quant_pre_process(
            input_model_path  = str(self.onnx_fp32_path),
            output_model_path = str(preprocessed_path),
        )

        quantize_dynamic(
            model_input  = str(preprocessed_path),
            model_output = str(self.onnx_int8_path),
            weight_type  = QuantType.QInt8,
        )
        # Remove temp file
        preprocessed_path.unlink(missing_ok=True)

        size_kb = self.onnx_int8_path.stat().st_size / 1024
        print(f"       INT8 ONNX → {self.onnx_int8_path}  ({size_kb:.1f} KB)")
        return self.onnx_int8_path

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Generate C weight-array header for firmware
    # ─────────────────────────────────────────────────────────────────────────

    def export_c_header(self) -> Path:
        """Extract float32 weights from checkpoint and write autoencoder_model.h."""
        from ml.tinyml.firmware_export.generate_header import generate_c_header

        print("  [3/3] Exporting weights → C header …")
        header_path = generate_c_header(
            checkpoint_path = self.checkpoint_path,
        )
        print(f"       C header  → {header_path}")
        return header_path

    # ─────────────────────────────────────────────────────────────────────────
    # Orchestrator
    # ─────────────────────────────────────────────────────────────────────────

    def run_full_pipeline(self) -> Path:
        """
        Execute all three conversion steps.

        Returns:
            Path to the INT8 ONNX file (used as tflite_path for compatibility).
        """
        print("\n" + "=" * 55)
        print("  Synera 2.0 — Quantization Pipeline")
        print("=" * 55)
        self.pytorch_to_onnx()
        self.onnx_to_int8()
        self.export_c_header()
        print("\n  Pipeline complete ✓")
        print(f"  INT8 model : {self.onnx_int8_path}")
        return self.onnx_int8_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize Synera autoencoder to INT8 TFLite")
    parser.add_argument(
        "--checkpoint",
        default="ml/tinyml/checkpoints/autoencoder_final.pt",
        help="Path to trained PyTorch checkpoint (relative to repo root)",
    )
    parser.add_argument(
        "--output-dir",
        default="ml/tinyml/checkpoints",
    )
    args = parser.parse_args()

    quantizer = ModelQuantizer(args.checkpoint, args.output_dir)
    quantizer.run_full_pipeline()
