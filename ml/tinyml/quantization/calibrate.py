"""
Synera 2.0 — Quantization Calibration Data
Wraps prepare_calibration_data() into a TFLite-compatible
representative dataset generator for INT8 quantization.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import numpy as np

# ── repo-root bootstrap ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]   # ml/tinyml/quantization → ml/tinyml → ml → ArogyaLink
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ─────────────────────────────────────────────────────────────────────────────


def get_representative_dataset(
    n_samples: int = 200,
) -> "Generator[list[np.ndarray], None, None]":
    """
    TFLite representative dataset generator.

    Yields single-sample float32 arrays of shape (1, 10, 5) for TFLite
    INT8 calibration.

    Usage (inside TFLite converter):
        converter.representative_dataset = get_representative_dataset
    """
    from scripts.data_gen.scenario_generator import prepare_calibration_data

    windows = prepare_calibration_data(n_windows=n_samples)   # (N, 10, 5)
    np.random.default_rng(42).shuffle(windows)

    def _generator():
        for i in range(len(windows)):
            # TFLite expects a list of input arrays, each (1, seq, features)
            sample = windows[i : i + 1].astype(np.float32)     # (1, 10, 5)
            yield [sample]

    return _generator


def save_calibration_windows(
    out_path: str | Path = "ml/tinyml/checkpoints/calibration_data.npy",
    n_samples: int = 200,
) -> Path:
    """
    Serialise calibration windows to disk for reproducibility.

    Returns:
        Path where the .npy file was saved.
    """
    from scripts.data_gen.scenario_generator import prepare_calibration_data

    out_path = Path(_ROOT) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    windows = prepare_calibration_data(n_windows=n_samples)
    np.save(out_path, windows)
    print(f"  Saved {len(windows)} calibration windows → {out_path}")
    return out_path
