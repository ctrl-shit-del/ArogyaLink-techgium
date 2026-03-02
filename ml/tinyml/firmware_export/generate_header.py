"""
Synera 2.0 — C Weight-Array Header Generator
Extracts float32 weights from a trained VitalAutoencoder PyTorch checkpoint
and writes autoencoder_model.h for the ESP32-S3 custom LSTM inference engine.

No TensorFlow or TFLite required.

Output (firmware/src/tinyml/autoencoder_model.h):
    - Architecture constants (SYNERA_HIDDEN_SIZE, etc.)
    - SYNERA_ANOMALY_THRESHOLD
    - C float32 arrays for every LSTM / Linear weight tensor

Usage:
    python ml/tinyml/firmware_export/generate_header.py \\
        [--checkpoint ml/tinyml/checkpoints/autoencoder_final.pt] \\
        [--header-out firmware/src/tinyml/autoencoder_model.h]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# ── repo-root bootstrap ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]   # ml/tinyml/firmware_export → ml/tinyml → ml → ArogyaLink
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

# Default paths
_DEFAULT_CHECKPOINT = _ROOT / "ml" / "tinyml" / "checkpoints" / "autoencoder_final.pt"
_HEADER_OUT         = _ROOT / "firmware" / "src" / "tinyml" / "autoencoder_model.h"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_array_1d(name: str, data: "np.ndarray") -> str:
    """Format a 1-D float array as a C definition."""
    import numpy as np
    vals = data.flatten().tolist()
    n    = len(vals)
    cols = 8
    rows = []
    for i in range(0, n, cols):
        chunk = vals[i : i + cols]
        rows.append("    " + ", ".join(f"{v:.8f}f" for v in chunk) + ",")
    return f"const float {name}[{n}] = {{\n" + "\n".join(rows) + "\n};\n"


def _fmt_array_2d(name: str, data: "np.ndarray") -> str:
    """Format a 2-D float array as a C definition (row-major)."""
    import numpy as np
    rows_n, cols_n = data.shape
    lines = [f"const float {name}[{rows_n}][{cols_n}] = {{"]
    for r in range(rows_n):
        row_vals = data[r].tolist()
        row_str  = ", ".join(f"{v:.8f}f" for v in row_vals)
        lines.append(f"    {{{row_str}}},")
    lines.append("};\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_c_header(
    checkpoint_path: "str | Path" = _DEFAULT_CHECKPOINT,
    header_path:     "str | Path | None" = None,
) -> Path:
    """
    Extract float32 weights from a VitalAutoencoder checkpoint and write the
    C weight-array header used by firmware/src/tinyml/model_runner.c.

    Args:
        checkpoint_path: Path to autoencoder_final.pt (absolute or relative to _ROOT).
        header_path:     Output .h path (default: firmware/src/tinyml/autoencoder_model.h).

    Returns:
        Path to the generated .h file.
    """
    import numpy as np
    import torch
    from ml.tinyml.model.autoencoder import AutoencoderConfig, VitalAutoencoder

    # Resolve paths ────────────────────────────────────────────────────────────
    cp_path     = Path(checkpoint_path) if Path(checkpoint_path).is_absolute() else Path(_ROOT) / checkpoint_path
    header_path = Path(header_path) if header_path else _HEADER_OUT
    header_path.parent.mkdir(parents=True, exist_ok=True)

    if not cp_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cp_path}")

    # Load model ───────────────────────────────────────────────────────────────
    model     = VitalAutoencoder.load(cp_path)
    # anomaly_threshold is stored on model.config; set as top-level attr too
    threshold = getattr(model, "anomaly_threshold", None) or model.config.anomaly_threshold
    state     = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    # Weight tensors ───────────────────────────────────────────────────────────
    # Encoder LSTM:   weight_ih (64,5), weight_hh (64,16), bias_ih (64), bias_hh (64)
    # Encoder FC:     weight (8,16), bias (8)
    # Decoder expand: weight (16,8), bias (16)
    # Decoder LSTM:   weight_ih (64,16), weight_hh (64,16), bias_ih (64), bias_hh (64)
    # Decoder out:    weight (5,16), bias (5)

    enc_lstm_wih = state["encoder.lstm.weight_ih_l0"]   # (64, 5)
    enc_lstm_whh = state["encoder.lstm.weight_hh_l0"]   # (64, 16)
    enc_lstm_bih = state["encoder.lstm.bias_ih_l0"]     # (64,)
    enc_lstm_bhh = state["encoder.lstm.bias_hh_l0"]     # (64,)

    enc_fc_w     = state["encoder.fc_latent.weight"]    # (8, 16)
    enc_fc_b     = state["encoder.fc_latent.bias"]      # (8,)

    dec_exp_w    = state["decoder.fc_expand.weight"]    # (16, 8)
    dec_exp_b    = state["decoder.fc_expand.bias"]      # (16,)

    dec_lstm_wih = state["decoder.lstm.weight_ih_l0"]   # (64, 16)
    dec_lstm_whh = state["decoder.lstm.weight_hh_l0"]   # (64, 16)
    dec_lstm_bih = state["decoder.lstm.bias_ih_l0"]     # (64,)
    dec_lstm_bhh = state["decoder.lstm.bias_hh_l0"]     # (64,)

    dec_out_w    = state["decoder.fc_out.weight"]       # (5, 16)
    dec_out_b    = state["decoder.fc_out.bias"]         # (5,)

    total_params = sum(
        a.size for a in [
            enc_lstm_wih, enc_lstm_whh, enc_lstm_bih, enc_lstm_bhh,
            enc_fc_w, enc_fc_b,
            dec_exp_w, dec_exp_b,
            dec_lstm_wih, dec_lstm_whh, dec_lstm_bih, dec_lstm_bhh,
            dec_out_w, dec_out_b,
        ]
    )
    total_bytes_fp32 = total_params * 4

    # Build C source ───────────────────────────────────────────────────────────
    sections = []

    sections.append(f"""\
#pragma once
// ============================================================
//  Auto-generated by Synera 2.0 firmware export pipeline
//  Checkpoint : {cp_path.name}
//  Parameters : {total_params} floats ({total_bytes_fp32 / 1024:.1f} KB FP32)
//  DO NOT EDIT — regenerate via:
//    python ml/tinyml/firmware_export/generate_header.py
// ============================================================

#include <stdint.h>

// ── Architecture constants ───────────────────────────────────────────────────
#define SYNERA_SEQ_LEN             10
#define SYNERA_FEATURES            5
#define SYNERA_HIDDEN_SIZE         16
#define SYNERA_LATENT_SIZE         8

// ── Anomaly threshold (μ + 2.5σ over normal validation set) ──────────────────
#define SYNERA_ANOMALY_THRESHOLD   {threshold:.5f}f
#define SYNERA_SYNERA_MSE_MULT     2.0f    // ×threshold → SYNERA STATE
""")

    sections.append("// ── Encoder LSTM: LSTM(input=5, hidden=16) ──────────────────────────────────\n")
    sections.append("// Weight gate order: input(i) | forget(f) | cell(g) | output(o)\n")
    sections.append(_fmt_array_2d("enc_lstm_wih", enc_lstm_wih))  # (64,5)
    sections.append(_fmt_array_2d("enc_lstm_whh", enc_lstm_whh))  # (64,16)
    sections.append(_fmt_array_1d("enc_lstm_bih", enc_lstm_bih))  # (64,)
    sections.append(_fmt_array_1d("enc_lstm_bhh", enc_lstm_bhh))  # (64,)

    sections.append("// ── Encoder FC: Linear(16→8) ─────────────────────────────────────────────────\n")
    sections.append(_fmt_array_2d("enc_fc_w", enc_fc_w))          # (8,16)
    sections.append(_fmt_array_1d("enc_fc_b", enc_fc_b))          # (8,)

    sections.append("// ── Decoder expand: Linear(8→16) ─────────────────────────────────────────────\n")
    sections.append(_fmt_array_2d("dec_exp_w", dec_exp_w))        # (16,8)
    sections.append(_fmt_array_1d("dec_exp_b", dec_exp_b))        # (16,)

    sections.append("// ── Decoder LSTM: LSTM(input=16, hidden=16) ─────────────────────────────────\n")
    sections.append(_fmt_array_2d("dec_lstm_wih", dec_lstm_wih))  # (64,16)
    sections.append(_fmt_array_2d("dec_lstm_whh", dec_lstm_whh))  # (64,16)
    sections.append(_fmt_array_1d("dec_lstm_bih", dec_lstm_bih))  # (64,)
    sections.append(_fmt_array_1d("dec_lstm_bhh", dec_lstm_bhh))  # (64,)

    sections.append("// ── Decoder out: Linear(16→5) ────────────────────────────────────────────────\n")
    sections.append(_fmt_array_2d("dec_out_w", dec_out_w))        # (5,16)
    sections.append(_fmt_array_1d("dec_out_b", dec_out_b))        # (5,)

    header_path.write_text("\n".join(sections), encoding="utf-8")

    size_kb = header_path.stat().st_size / 1024
    print(f"  C header generated → {header_path}")
    print(f"  Weights: {total_params} params  |  FP32: {total_bytes_fp32 / 1024:.1f} KB  |  Header: {size_kb:.1f} KB")
    return header_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate C weight-array header from checkpoint")
    parser.add_argument(
        "--checkpoint",
        default="ml/tinyml/checkpoints/autoencoder_final.pt",
        help="Path to checkpoint (relative to repo root or absolute)",
    )
    parser.add_argument("--header-out", default=None, help="Output .h path")
    args = parser.parse_args()

    out = generate_c_header(
        checkpoint_path = args.checkpoint,
        header_path     = args.header_out,
    )
    print(f"  Done: {out}")
