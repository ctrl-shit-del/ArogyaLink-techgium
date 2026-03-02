"""
Synera 2.0 — Firmware Export Orchestrator
Drives the full PyTorch → INT8 ONNX + C weight-array header pipeline.
This is the single script to run after training is complete.

No TensorFlow required.  Pipeline:
    1. PyTorch → FP32 ONNX          (torch.onnx.export)
    2. FP32 ONNX → INT8 ONNX        (onnxruntime dynamic quantization)
    3. checkpoint → C header         (float32 weight arrays for model_runner.c)
    4. Validate accuracy drop        (optional, onnxruntime inference)
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── repo-root bootstrap ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]   # ml/tinyml/firmware_export → ml/tinyml → ml → ArogyaLink
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ─────────────────────────────────────────────────────────────────────────────


def convert_to_tflite(
    checkpoint_path: str | Path = "ml/tinyml/checkpoints/autoencoder_final.pt",
    output_dir:      str | Path = "ml/tinyml/checkpoints",
    validate:        bool        = True,
    generate_header: bool        = True,
) -> Path:
    """
    Full firmware export pipeline:
      1. PyTorch checkpoint → FP32 ONNX
      2. FP32 ONNX → INT8 ONNX (onnxruntime dynamic quantization)
      3. Checkpoint → C weight-array header  (optional)
      4. Validate MSE accuracy drop          (optional)

    Returns:
        Path to the INT8 ONNX file (autoencoder_int8.onnx).
    """
    from ml.tinyml.quantization.quantize import ModelQuantizer

    quantizer   = ModelQuantizer(checkpoint_path, output_dir)
    onnx_int8   = quantizer.run_full_pipeline()

    if validate:
        try:
            from ml.tinyml.quantization.validate import verify_quantization
            results = verify_quantization(
                pytorch_checkpoint = checkpoint_path,
                tflite_path        = onnx_int8.relative_to(_ROOT),
            )
            if not results.get("all_pass", True):
                print("  WARNING: Quantization gate failed — check validate.py output above.")
        except Exception as e:
            print(f"  [validate skipped: {e}]")

    if generate_header:
        from ml.tinyml.firmware_export.generate_header import generate_c_header
        generate_c_header(checkpoint_path=checkpoint_path)

    return onnx_int8


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export trained Synera model to INT8 ONNX + C header")
    parser.add_argument("--checkpoint",       default="ml/tinyml/checkpoints/autoencoder_final.pt")
    parser.add_argument("--output-dir",       default="ml/tinyml/checkpoints")
    parser.add_argument("--skip-validate",    action="store_true")
    parser.add_argument("--skip-header",      action="store_true")
    args = parser.parse_args()

    out = convert_to_tflite(
        checkpoint_path = args.checkpoint,
        output_dir      = args.output_dir,
        validate        = not args.skip_validate,
        generate_header = not args.skip_header,
    )
    print(f"\n  INT8 model : {out}")
