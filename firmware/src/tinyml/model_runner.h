/**
 * Synera 2.0 — Custom LSTM Autoencoder Inference Engine
 * ======================================================
 * Minimal, self-contained LSTM autoencoder for anomaly detection on ESP32-S3.
 * No third-party ML library required — uses float32 weights from autoencoder_model.h.
 *
 * Architecture (mirrors VitalAutoencoder in Python):
 *   Input  : float[SYNERA_SEQ_LEN][SYNERA_FEATURES]  (normalised vital-sign window)
 *   Encoder: LSTM(5→16) → Linear(16→8) → latent[8]
 *   Decoder: Linear(8→16) → tile×10 → LSTM(16→16) → Linear(16→5) → sigmoid → recon[10][5]
 *   MSE    : mean squared error over all 50 elements
 *
 * Usage:
 *   #include "model_runner.h"
 *
 *   float window[SYNERA_SEQ_LEN][SYNERA_FEATURES];
 *   // ... fill window with normalised vital signs ...
 *
 *   float mse   = synera_infer(window);
 *   bool  alert = synera_is_anomaly(mse);
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "autoencoder_model.h"   /* generated weight arrays + SYNERA_ANOMALY_THRESHOLD */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Run a full encoder–decoder pass on a 10×5 vital-sign window.
 *
 * @param input  Pointer to a [SYNERA_SEQ_LEN][SYNERA_FEATURES] float array.
 *               Values must be normalised to [0, 1] (same normalisation used during training).
 *
 * @return Mean squared reconstruction error (MSE).
 *         - Low MSE  → pattern matches training distribution → NORMAL
 *         - High MSE → anomalous trajectory → PRE-ALERT or SYNERA STATE
 */
float synera_infer(const float input[SYNERA_SEQ_LEN][SYNERA_FEATURES]);

/**
 * @brief Check whether an MSE value exceeds the calibrated anomaly threshold.
 *
 * @param mse  Reconstruction error returned by synera_infer().
 * @return     true if mse > SYNERA_ANOMALY_THRESHOLD (anomaly), false otherwise.
 */
bool synera_is_anomaly(float mse);

#ifdef __cplusplus
}
#endif
