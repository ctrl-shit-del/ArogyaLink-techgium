"""
Synera 2.0 — Patient Archetype Profile Configurations
Each profile defines the physiological baseline and trajectory behaviour
for a simulated patient type.  Used by PatientSimulator.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from ml.shared.constants import (
    ARCHETYPE_STABLE, ARCHETYPE_TRAJECTORY_ACCEL, ARCHETYPE_EXERTION,
    ARCHETYPE_ARTIFACT_GLITCH, ARCHETYPE_SLOW_DRIFT, ARCHETYPE_POST_OP_RECOVERY,
)


@dataclass
class ArchetypeProfile:
    name: str
    # ── Resting baseline ──────────────────────────────────────────────────────
    rest_hr_mean: float      # BPM
    rest_hr_std: float
    rest_spo2_mean: float    # %
    rest_spo2_std: float
    rest_temp_mean: float    # °C
    rest_temp_std: float
    rest_sys_bp: float       # mmHg
    rest_dia_bp: float       # mmHg
    rest_motion_mean: float  # 0–10
    rest_motion_std: float

    # ── Trajectory parameters ─────────────────────────────────────────────────
    # How many readings of stable data before any trajectory event starts
    stable_readings_before_event: int = 0
    # Type of HR trajectory after the stable phase
    # "none" | "exponential" | "linear" | "exertion_burst" | "glitch_spike"
    hr_trajectory: str = "none"
    # For exponential / linear trajectories
    trajectory_target_hr: float = 0.0     # final HR at end of window
    trajectory_duration_readings: int = 0  # how many readings the trajectory lasts
    # SpO2 drop during trajectory (absolute percentage drop from baseline)
    spo2_drop_total: float = 0.0
    # Temp rise during trajectory
    temp_rise_total: float = 0.0
    # Motion behaviour during trajectory
    motion_high_during_event: bool = False   # True for exertion archetype
    exertion_motion_level: float = 8.0

    # ── Glitch parameters ─────────────────────────────────────────────────────
    glitch_interval_min: int = 60   # min readings between glitches
    glitch_interval_max: int = 120
    glitch_hr_spike: float = 220.0  # BPM during glitch
    glitch_spo2_drop: float = 65.0  # % during glitch

    # ── Description for display ───────────────────────────────────────────────
    description: str = ""
    expected_alert: str = "none"  # "none" | "synera_state" | "watch"


# ─────────────────────────────────────────────────────────────────────────────
# Profile registry
# ─────────────────────────────────────────────────────────────────────────────

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
        # 20 minutes stable at 5s = 240 readings, then deterioration begins
        stable_readings_before_event=240,
        hr_trajectory="exponential",
        trajectory_target_hr=185.0,
        trajectory_duration_readings=120,   # 10 minutes of acceleration
        spo2_drop_total=9.0,                # 97 → 88
        temp_rise_total=0.9,                # 36.8 → 37.7
        motion_high_during_event=False,     # CRITICAL: at rest throughout
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
        trajectory_duration_readings=36,    # 3-minute bathroom walk
        spo2_drop_total=2.0,                # trivial drop, recovers quickly
        motion_high_during_event=True,      # HIGH motion — this is the key filter
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
        stable_readings_before_event=0,
        hr_trajectory="linear",
        trajectory_target_hr=95.0,
        # 45 min at 5s = 540 readings for linear drift
        trajectory_duration_readings=540,
        spo2_drop_total=1.5,
        motion_high_during_event=False,
        description="HR drifts linearly over 45 min. Constant first derivative. WATCH tier, not CRITICAL.",
        expected_alert="watch",
    ),

    ARCHETYPE_POST_OP_RECOVERY: ArchetypeProfile(
        name=ARCHETYPE_POST_OP_RECOVERY,
        rest_hr_mean=95.0, rest_hr_std=6.0,    # elevated baseline post-surgery
        rest_spo2_mean=95.0, rest_spo2_std=1.0,
        rest_temp_mean=37.4, rest_temp_std=0.25,
        rest_sys_bp=140.0, rest_dia_bp=88.0,
        rest_motion_mean=1.0, rest_motion_std=1.2,  # repositioning noise
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
