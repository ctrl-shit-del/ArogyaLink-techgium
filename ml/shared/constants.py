"""
Synera 2.0 — Shared Constants
Feature ranges, normalization bounds, and decision thresholds.
This is the single source of truth for all feature definitions.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Feature order (DO NOT CHANGE — matches firmware payload and MQTT contract)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = ["heart_rate", "spo2", "temperature", "estimated_bp", "motion_score"]
N_FEATURES = 5
SEQUENCE_LENGTH = 10          # 10 readings × 5s = 50-second sliding window
SAMPLING_INTERVAL_SEC = 5     # ESP32-S3 reads every 5 seconds

# ─────────────────────────────────────────────────────────────────────────────
# Raw value ranges (physiologically valid bounds)
# Values outside these bounds are treated as hardware artifacts (Rule A)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_RANGES = {
    "heart_rate":      {"min": 30.0,   "max": 220.0, "unit": "BPM"},
    "spo2":            {"min": 85.0,   "max": 100.0, "unit": "%"},
    "temperature":     {"min": 35.0,   "max": 42.0,  "unit": "°C"},
    "estimated_bp":    {"min": 0.0,    "max": 1.0,   "unit": "score"},  # already normalized composite
    "motion_score":    {"min": 0.0,    "max": 10.0,  "unit": "score"},
}

# Raw systolic / diastolic ranges (used to build estimated_bp composite)
SYS_BP_MIN, SYS_BP_MAX = 70.0, 200.0
DIA_BP_MIN, DIA_BP_MAX = 40.0, 120.0

# ─────────────────────────────────────────────────────────────────────────────
# Rule A — Artifact rejection delta thresholds (per reading, 5-second window)
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACT_DELTA = {
    "heart_rate":   40.0,   # BPM — human physiology cannot change HR by 40 BPM in 5s
    "spo2":          5.0,   # %   — cannot drop 5% in one reading without artifact
    "temperature":   0.5,   # °C  — thermal inertia makes >0.5°C impossible in 5s
}

# ─────────────────────────────────────────────────────────────────────────────
# Rule B — Exertion filter (motion gating)
# ─────────────────────────────────────────────────────────────────────────────
EXERTION_MOTION_THRESHOLD = 4.0   # motion_score > 4 → patient is active
EXERTION_HR_ELEVATION_BPM = 15.0  # HR must be elevated by this much to classify as exertion

# ─────────────────────────────────────────────────────────────────────────────
# Rule C — Trajectory acceleration detection
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_DEVIATION_SIGMA = 1.5    # z-score threshold for "significant deviation"
AT_REST_MOTION_MAX = 2.0          # motion_score < 2 → patient is at rest
ACCELERATION_SUSTAIN_READINGS = 3 # second derivative must be positive for this many consecutive readings

# ─────────────────────────────────────────────────────────────────────────────
# TinyML thresholds — set by training pipeline, defaults here for reference
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ANOMALY_THRESHOLD = 0.035  # reconstruction error > this = pre_alert=True
TINYML_MODEL_MAX_SIZE_KB  = 45.0   # ESP32-S3 constraint
TINYML_RAM_MAX_KB         = 50.0   # inference RAM budget

# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────
CALIBRATION_WINDOW_READINGS = 360         # 30 minutes × 12 readings/min
CALIBRATION_MIN_READINGS    = 36          # absolute minimum (3 minutes) before partial calibration
ROLLING_UPDATE_ALPHA        = 0.001       # EMA alpha for post-calibration baseline drift

# ─────────────────────────────────────────────────────────────────────────────
# Correlated noise covariance matrix (used by simulator)
# Row/col order: [HR, SpO2, Temp, BP_score, Motion]
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np

# Built from correlation matrix and std-dev diagonal so the matrix is
# guaranteed symmetric positive-semidefinite:
#   stds  : HR=2.0, SpO2=0.50, Temp=0.10, BP=1.00, Motion=1.41
#   corrs : HR↔SpO2=-0.30, HR↔BP=+0.50, SpO2↔BP=-0.20  (all |r|<1)
PHYSIOLOGICAL_NOISE_COV = np.array([
    [ 4.000, -0.300,  0.020,  1.000,  0.140],   # HR
    [-0.300,  0.250,  0.005, -0.100, -0.035],   # SpO2
    [ 0.020,  0.005,  0.010,  0.010,  0.001],   # Temperature
    [ 1.000, -0.100,  0.010,  1.000,  0.070],   # BP (correlated with HR)
    [ 0.140, -0.035,  0.001,  0.070,  2.000],   # Motion (mostly independent)
], dtype=np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# Patient archetypes
# ─────────────────────────────────────────────────────────────────────────────
ARCHETYPE_STABLE            = "STABLE"
ARCHETYPE_TRAJECTORY_ACCEL  = "TRAJECTORY_ACCEL"
ARCHETYPE_EXERTION          = "EXERTION"
ARCHETYPE_ARTIFACT_GLITCH   = "ARTIFACT_GLITCH"
ARCHETYPE_SLOW_DRIFT        = "SLOW_DRIFT"
ARCHETYPE_POST_OP_RECOVERY  = "POST_OP_RECOVERY"

ALL_ARCHETYPES = [
    ARCHETYPE_STABLE,
    ARCHETYPE_TRAJECTORY_ACCEL,
    ARCHETYPE_EXERTION,
    ARCHETYPE_ARTIFACT_GLITCH,
    ARCHETYPE_SLOW_DRIFT,
    ARCHETYPE_POST_OP_RECOVERY,
]

# ─────────────────────────────────────────────────────────────────────────────
# Priority tier labels
# ─────────────────────────────────────────────────────────────────────────────
TIER_CRITICAL = "CRITICAL"    # Synera State
TIER_WATCH    = "WATCH"
TIER_STABLE   = "STABLE"
