"""Five mock patient profiles for simulator and demo.

Also exposes ArchetypeProfile, PROFILES, and get_profile for use by
PatientSimulator (demo_scenarios.py and training pipelines).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.shared.constants import (
    ARCHETYPE_STABLE, ARCHETYPE_TRAJECTORY_ACCEL, ARCHETYPE_EXERTION,
    ARCHETYPE_ARTIFACT_GLITCH, ARCHETYPE_SLOW_DRIFT, ARCHETYPE_POST_OP_RECOVERY,
)

PATIENT_PROFILES: dict[str, dict[str, Any]] = {
    "PT-0001": {
        "name": "Rajesh Kumar",
        "age": 62,
        "gender": "Male",
        "scenario": "STABLE",
        "ward": "General Ward A",
        "bed": "12B",
        "conditions": [
            {"code": "I10", "name": "Hypertension"},
            {"code": "E11", "name": "Type 2 Diabetes"},
        ],
        "medications": [
            {"name": "Metformin", "dose": "500mg", "frequency": "BD"},
            {"name": "Amlodipine", "dose": "5mg", "frequency": "OD"},
        ],
        "baseline_hr": 88,
        "baseline_spo2": 96,
        "baseline_temp": 37.0,
        "trajectory": "flat_with_noise",
        "expected_outcome": "STABLE — no alert ever",
    },
    "PT-0002": {
        "name": "Priya Sharma",
        "age": 34,
        "gender": "Female",
        "scenario": "TRAJECTORY_ACCELERATION",
        "ward": "Surgical Ward B",
        "bed": "4A",
        "conditions": [
            {"code": "Z48.8", "name": "Post-appendectomy Day 2"},
            {"code": "E11", "name": "Type 2 Diabetes"},
        ],
        "medications": [
            {"name": "Cefazolin", "dose": "1g IV", "frequency": "Q8H"},
            {"name": "Metformin", "dose": "500mg", "frequency": "BD"},
            {"name": "Paracetamol", "dose": "500mg", "frequency": "Q6H PRN"},
        ],
        "baseline_hr": 82,
        "baseline_spo2": 97,
        "baseline_temp": 37.0,
        "trajectory": "exponential_acceleration",
        "expected_outcome": "SYNERA_STATE fired at ~103 BPM (8-15 min before threshold)",
    },
    "PT-0003": {
        "name": "Arjun Mehta",
        "age": 28,
        "gender": "Male",
        "scenario": "EXERTION_FALSE_POSITIVE",
        "ward": "General Ward A",
        "bed": "7C",
        "conditions": [],
        "medications": [],
        "baseline_hr": 68,
        "baseline_spo2": 99,
        "baseline_temp": 37.0,
        "trajectory": "exertion_spike",
        "expected_outcome": "EXERTION_LOGGED — NO alert. Rule B catches this.",
    },
    "PT-0004": {
        "name": "Fatima Begum",
        "age": 71,
        "gender": "Female",
        "scenario": "GLITCH_SPIKE",
        "ward": "ICU Step-Down",
        "bed": "2",
        "conditions": [
            {"code": "J44", "name": "COPD"},
            {"code": "I48", "name": "Atrial Fibrillation"},
        ],
        "medications": [
            {"name": "Warfarin", "dose": "5mg", "frequency": "OD"},
            {"name": "Salbutamol", "dose": "100mcg inhaler", "frequency": "PRN"},
            {"name": "Digoxin", "dose": "0.25mg", "frequency": "OD"},
        ],
        "baseline_hr": 79,
        "baseline_spo2": 94,
        "baseline_temp": 37.0,
        "trajectory": "glitch_spike",
        "glitch_value": 228,
        "glitch_index": 3,
        "expected_outcome": "ARTIFACT — packet silently discarded.",
    },
    "PT-0005": {
        "name": "Suresh Patel",
        "age": 55,
        "gender": "Male",
        "scenario": "SLOW_DRIFT_WATCH",
        "ward": "General Ward C",
        "bed": "9A",
        "conditions": [
            {"code": "I10", "name": "Hypertension"},
            {"code": "N18.3", "name": "CKD Stage 3"},
        ],
        "medications": [
            {"name": "Losartan", "dose": "50mg", "frequency": "OD"},
            {"name": "Furosemide", "dose": "40mg", "frequency": "OD"},
        ],
        "baseline_hr": 76,
        "baseline_spo2": 97,
        "baseline_temp": 37.0,
        "trajectory": "slow_linear_drift",
        "expected_outcome": "WATCH tier — deviation triggered, no acceleration, no RAG call",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Archetype Profiles (used by PatientSimulator and demo_scenarios)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArchetypeProfile:
    name: str
    # ── Resting baseline ──────────────────────────────────────────────────────
    rest_hr_mean: float
    rest_hr_std: float
    rest_spo2_mean: float
    rest_spo2_std: float
    rest_temp_mean: float
    rest_temp_std: float
    rest_sys_bp: float
    rest_dia_bp: float
    rest_motion_mean: float
    rest_motion_std: float

    # ── Trajectory parameters ─────────────────────────────────────────────────
    stable_readings_before_event: int = 0
    hr_trajectory: str = "none"
    trajectory_target_hr: float = 0.0
    trajectory_duration_readings: int = 0
    spo2_drop_total: float = 0.0
    temp_rise_total: float = 0.0
    motion_high_during_event: bool = False
    exertion_motion_level: float = 8.0

    # ── Glitch parameters ─────────────────────────────────────────────────────
    glitch_interval_min: int = 60
    glitch_interval_max: int = 120
    glitch_hr_spike: float = 220.0
    glitch_spo2_drop: float = 65.0

    # ── Description ───────────────────────────────────────────────────────────
    description: str = ""
    expected_alert: str = "none"


PROFILES: dict[str, ArchetypeProfile] = {

    ARCHETYPE_STABLE: ArchetypeProfile(
        name=ARCHETYPE_STABLE,
        rest_hr_mean=72.0, rest_hr_std=5.0,
        rest_spo2_mean=97.0, rest_spo2_std=1.0,
        rest_temp_mean=36.8, rest_temp_std=0.2,
        rest_sys_bp=118.0, rest_dia_bp=76.0,
        rest_motion_mean=0.5, rest_motion_std=0.8,
        stable_readings_before_event=9999,
        hr_trajectory="none",
        description="Healthy resting adult — ward patient, no acute condition.",
        expected_alert="none",
    ),

    ARCHETYPE_TRAJECTORY_ACCEL: ArchetypeProfile(
        name=ARCHETYPE_TRAJECTORY_ACCEL,
        rest_hr_mean=80.0, rest_hr_std=4.0,
        rest_spo2_mean=97.0, rest_spo2_std=0.8,
        rest_temp_mean=36.8, rest_temp_std=0.15,
        rest_sys_bp=120.0, rest_dia_bp=80.0,
        rest_motion_mean=0.3, rest_motion_std=0.4,
        stable_readings_before_event=240,
        hr_trajectory="exponential",
        trajectory_target_hr=185.0,
        trajectory_duration_readings=120,
        spo2_drop_total=9.0,
        temp_rise_total=0.9,
        motion_high_during_event=False,
        description="Post-op patient. Silent deterioration onset (early sepsis / PE).",
        expected_alert="synera_state",
    ),

    ARCHETYPE_EXERTION: ArchetypeProfile(
        name=ARCHETYPE_EXERTION,
        rest_hr_mean=72.0, rest_hr_std=4.0,
        rest_spo2_mean=97.0, rest_spo2_std=0.8,
        rest_temp_mean=36.7, rest_temp_std=0.15,
        rest_sys_bp=116.0, rest_dia_bp=74.0,
        rest_motion_mean=0.4, rest_motion_std=0.6,
        stable_readings_before_event=60,
        hr_trajectory="exertion_burst",
        trajectory_target_hr=132.0,
        trajectory_duration_readings=36,
        spo2_drop_total=2.0,
        motion_high_during_event=True,
        exertion_motion_level=8.0,
        description="Ward patient walks to bathroom. High motion + HR spike. Should NOT alert.",
        expected_alert="none",
    ),

    ARCHETYPE_ARTIFACT_GLITCH: ArchetypeProfile(
        name=ARCHETYPE_ARTIFACT_GLITCH,
        rest_hr_mean=75.0, rest_hr_std=4.0,
        rest_spo2_mean=97.0, rest_spo2_std=0.8,
        rest_temp_mean=36.9, rest_temp_std=0.2,
        rest_sys_bp=122.0, rest_dia_bp=80.0,
        rest_motion_mean=0.3, rest_motion_std=0.5,
        hr_trajectory="glitch_spike",
        glitch_interval_min=60,
        glitch_interval_max=120,
        glitch_hr_spike=228.0,
        glitch_spo2_drop=63.0,
        description="Stable patient with loose PPG sensor causing periodic artifact spikes.",
        expected_alert="none",
    ),

    ARCHETYPE_SLOW_DRIFT: ArchetypeProfile(
        name=ARCHETYPE_SLOW_DRIFT,
        rest_hr_mean=72.0, rest_hr_std=3.0,
        rest_spo2_mean=96.5, rest_spo2_std=0.8,
        rest_temp_mean=37.0, rest_temp_std=0.15,
        rest_sys_bp=125.0, rest_dia_bp=82.0,
        rest_motion_mean=0.4, rest_motion_std=0.5,
        stable_readings_before_event=15,
        hr_trajectory="exponential",
        trajectory_target_hr=155.0,
        trajectory_duration_readings=100,
        spo2_drop_total=1.5,
        motion_high_during_event=False,
        description="HR drifts linearly over 45 min. WATCH tier, not CRITICAL.",
        expected_alert="watch",
    ),

    ARCHETYPE_POST_OP_RECOVERY: ArchetypeProfile(
        name=ARCHETYPE_POST_OP_RECOVERY,
        rest_hr_mean=95.0, rest_hr_std=6.0,
        rest_spo2_mean=95.0, rest_spo2_std=1.0,
        rest_temp_mean=37.4, rest_temp_std=0.25,
        rest_sys_bp=140.0, rest_dia_bp=88.0,
        rest_motion_mean=1.0, rest_motion_std=1.2,
        hr_trajectory="none",
        description="Post-op patient. Elevated but stable vitals. Synera calibrates to their new normal.",
        expected_alert="none",
    ),
}


def get_profile(archetype: str) -> ArchetypeProfile:
    """Retrieve archetype profile by name.  Raises KeyError on invalid archetype."""
    if archetype not in PROFILES:
        raise KeyError(f"Unknown archetype '{archetype}'. Valid: {list(PROFILES.keys())}")
    return PROFILES[archetype]
