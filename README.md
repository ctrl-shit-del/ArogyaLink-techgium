# ArogyaLink — Synera 2.0

Continuous vitals intelligence platform. Personalised anomaly detection on ESP32-S3 using a lightweight LSTM Autoencoder (4K params, ~4.4 KB INT8).

---

## TinyML — Training & Testing

### Prerequisites

```bash
pip install -r ml/requirements.txt
```

> Minimum: `torch>=2.0`, `numpy>=1.24`, `scipy>=1.10`, `rich>=13.0`

---

### 1. Quick sanity check (10 epochs, ~15 seconds)

Verifies the full pipeline end-to-end without needing a GPU or large dataset.

```bash
python -m ml.tinyml.training.trainer --quick-demo
```

Expected output:
```
  Training complete in ~6s — best val loss: ~0.004
  Anomaly threshold (μ + 3σ): ~0.007
  Quick-demo PASSED
```

---

### 2. Full training (80 epochs, 50 patients × 60 min)

```bash
python -m ml.tinyml.training.trainer --epochs 80 --batch-size 64
```

Expected: early stopping typically triggers around epoch 21–46, ~1–3 minutes on CPU.
```
  Train windows : 30,218  Val windows : 5,332
  Early stop triggered at epoch 27 (patience=15)
  Training complete in ~95s — best val loss: 0.00048
  Anomaly threshold (μ + 2.5σ): 0.00174
  Curve saved → ml/tinyml/checkpoints/training_curves.png
```

> Threshold uses **2.5σ** (down from 3.0σ in earlier versions) giving ~2–4 extra
> readings of lead time at the cost of a theoretical FP rate rise from 0.13% → 0.62%
> (still near-zero in practice).

Saves three files to `ml/tinyml/checkpoints/`:
- `best_autoencoder.pt` — best checkpoint (lowest val loss)
- `autoencoder_final.pt` — final model with calibrated anomaly threshold
- `training_curves.png` — loss curves (always saved automatically)

---

### 3. Run the end-to-end simulation demo

When `autoencoder_final.pt` is present the demo runs in **LSTM mode** — the autoencoder's reconstruction error drives alert decisions and the MSE column shows real values. Without a checkpoint it falls back to rule-based mode automatically (no torch required).

```bash
# All three detection scenarios + calibration demo
python scripts/testing/demo_scenarios.py

# Individual flags
python scripts/testing/demo_scenarios.py --all-scenarios
python scripts/testing/demo_scenarios.py --calibration-demo
python scripts/testing/demo_scenarios.py --scenario silent_deterioration
python scripts/testing/demo_scenarios.py --scenario bathroom_walk
python scripts/testing/demo_scenarios.py --scenario glitch_rejection
```

Expected summary (LSTM mode, threshold 0.00174):
```
  Scenario                     Result  FP/stable   Alert rate
  ---------------------------- ------  ----------  ----------
  bathroom_walk                PASS    0              0.0%
  silent_deterioration         PASS    0             49.5%  (incl. 37 SYNERA STATE)
  glitch_rejection             PASS    0              0.0%
```

**Three-tier alert system:**

| Status | Condition | Meaning |
|---|---|---|
| `normal` | MSE < threshold | Vital pattern within learned baseline |
| `*** PRE-ALERT ***` | MSE > threshold | Autoencoder cannot reconstruct — anomaly signal |
| `!!! SYNERA STATE !!!` | MSE > threshold × 2.0 **AND** σ > 8.0 sustained 3+ readings | Immediate escalation |

Sample MSE output from `silent_deterioration` Phase 3:
```
    #      HR    SpO2   Temp  Motion   sigma      MSE   Status
   96    94    95.4%  37.0   0.9     7.73   0.0016   *** PRE-ALERT ***
  109   102    94.7%  37.0   0.4    10.71   0.0032   *** PRE-ALERT ***
  115   105    94.5%  37.1   0.9    11.07   0.0037   !!! SYNERA STATE !!!
  127   120    93.5%  37.2   0.4    14.03   0.0059   !!! SYNERA STATE !!!
```
Lead time vs static HR>120 threshold: **+2m 10s earlier**.
SYNERA STATE fires at reading #115 (HR=105 BPM) — well before the
static alarm would trigger. Reconstruction error crosses the learned
threshold of **0.00174** because the model never saw this deteriorating
vital pattern during training on healthy patients.

The calibration demo shows the core personalisation insight:
> Athlete (resting HR=52) and hypertensive (resting HR=90) both reach HR=95.
> Athlete fires PRE-ALERT (+11.6σ above personal baseline); hypertensive does not (+1.2σ).

---

### 4. Quantize to INT8 TFLite (requires tensorflow + onnx-tf)

```bash
# Full conversion: PyTorch → ONNX → TF SavedModel → INT8 TFLite
python ml/tinyml/firmware_export/convert_to_tflite.py \
    --checkpoint ml/tinyml/checkpoints/autoencoder_final.pt \
    --output-dir ml/tinyml/checkpoints

# Skip validation (faster)
python ml/tinyml/firmware_export/convert_to_tflite.py --skip-validate

# Skip C header generation
python ml/tinyml/firmware_export/convert_to_tflite.py --skip-header
```

Output:
```
ml/tinyml/checkpoints/autoencoder.onnx
ml/tinyml/checkpoints/autoencoder_savedmodel/
ml/tinyml/checkpoints/autoencoder_int8.tflite       (~4.4 KB)
firmware/src/tinyml/autoencoder_model.h             (C byte array)
```

---

### 5. Validate quantization accuracy (FP32 vs INT8)

```bash
python ml/tinyml/quantization/validate.py \
    --checkpoint ml/tinyml/checkpoints/autoencoder_final.pt \
    --tflite    ml/tinyml/checkpoints/autoencoder_int8.tflite \
    --n-eval 500
```

Deployment gate thresholds:
| Check | Limit |
|---|---|
| AUC-ROC drop (FP32 → INT8) | < 0.02 |
| Mean absolute MSE difference | < 0.005 |
| TFLite model size | < 50 KB |

---

### 6. Generate C header only (if .tflite already exists)

```bash
python ml/tinyml/firmware_export/generate_header.py \
    --tflite ml/tinyml/checkpoints/autoencoder_int8.tflite
# → firmware/src/tinyml/autoencoder_model.h
```

---

### 7. Generate data manually (optional)

```bash
# Inspect training set sizes
python scripts/data_gen/scenario_generator.py

# Generate calibration windows for INT8 quantization
python -c "
from scripts.data_gen.scenario_generator import prepare_calibration_data
import numpy as np
X = prepare_calibration_data(n_windows=200)
print(X.shape)   # (200, 10, 5)
"
```

---

## Model Architecture

```
Input  (batch, 10, 5)
  ↓  LSTMEncoder:  LSTM(5→16) + Linear(16→8)       1,608 params
Latent (batch, 8)
  ↓  LSTMDecoder:  Linear(8→16) + LSTM(16→16) + Linear(16→5) + Sigmoid
Output (batch, 10, 5)                               2,405 params
                                              Total: 4,013 params
                                              INT8:  ~4.4 KB
```

Feature vector: `[HR, SpO₂, Temperature, BP_composite, MotionScore]` — all normalised to `[0, 1]`.

Anomaly threshold is auto-calibrated as **μ + 2.5σ** of validation reconstruction errors after training.

---

## File Map

```
ml/
  shared/             constants, preprocessing, feature engineering
  baseline/           Welford online calibration, per-patient z-score baseline
  tinyml/
    model/            LSTM encoder, decoder, autoencoder
    training/         dataset, loss, callbacks, trainer
    quantization/     INT8 calibration, PyTorch→TFLite conversion, validation
    firmware_export/  TFLite orchestrator, C header generator
  requirements.txt

scripts/
  data_gen/           patient profiles, simulator, scenario generator
  testing/            demo_scenarios.py  ← start here
```
