"""
drl/state_builder.py
Converts a live alert context dict (from synera_engine) into a
17-dimensional numpy float32 state vector for the PPO agent.
"""

import numpy as np
from datetime import datetime, date, timezone
from typing import Dict, Any

RISK_TIER_MAP = {"Low": 0.1, "Medium": 0.4, "High": 0.7, "Critical": 1.0}
STATE_DIM = 17


def _age_from_patient_row(patient_row: Dict) -> int:
    """Compute age from patient row (dob or age)."""
    if patient_row.get("age") is not None:
        return int(patient_row["age"])
    dob = patient_row.get("dob")
    if not dob:
        return 40
    try:
        if isinstance(dob, str):
            d = date.fromisoformat(dob[:10])
        else:
            d = dob
        today = date.today()
        return today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    except Exception:
        return 40


def build_state_vector(alert_context: Dict[str, Any]) -> np.ndarray:
    """
    Convert alert context dict → numpy float32 array of shape (17,).
    All values are normalised to approximately [-1, 2] range.
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)
    vitals = alert_context.get("vitals", {})
    traj   = alert_context.get("trajectory", {})
    pat    = alert_context.get("patient", {})
    ctx    = alert_context.get("context", {})
    risk   = pat.get("genomic_risk", {})

    # Vital deviations in sigma units
    hr_std   = max(traj.get("hr_baseline_std", 5.0), 0.1)
    spo2_std = max(traj.get("spo2_baseline_std", 1.0), 0.1)
    temp_std = max(traj.get("temp_baseline_std", 0.3), 0.1)
    bp_std   = max(traj.get("bp_baseline_std", 8.0), 0.1)

    state[0] = (vitals.get("heart_rate", 80) - traj.get("hr_baseline_mean", 80)) / hr_std
    state[1] = (traj.get("spo2_baseline_mean", 97) - vitals.get("spo2", 97)) / spo2_std
    state[2] = (vitals.get("temperature", 37.0) - traj.get("temp_baseline_mean", 37.0)) / temp_std
    state[3] = (vitals.get("sys_bp_est", 120) - traj.get("bp_baseline_mean", 120)) / bp_std
    state[4] = float(vitals.get("motion_score", 0)) / 10.0

    # Trajectory signals (clip to prevent outliers)
    state[5] = float(np.clip(traj.get("hr_first_derivative", 0) / 20.0, -1, 1))
    state[6] = float(np.clip(traj.get("hr_second_derivative", 0) / 10.0, -1, 1))
    state[7] = float(np.clip(traj.get("spo2_second_derivative", 0) / 5.0, -1, 1))

    # Patient risk
    state[8]  = float(pat.get("age", 40)) / 100.0
    state[9]  = RISK_TIER_MAP.get(risk.get("cardiac", "Low"), 0.1)
    state[10] = RISK_TIER_MAP.get(risk.get("respiratory", "Low"), 0.1)
    state[11] = RISK_TIER_MAP.get(risk.get("diabetic", "Low"), 0.1)
    state[12] = RISK_TIER_MAP.get(risk.get("sepsis", "Low"), 0.1)

    # Cyclic time encoding
    hour = ctx.get("hour", datetime.now().hour)
    state[13] = float(np.sin(2 * np.pi * hour / 24))
    state[14] = float(np.cos(2 * np.pi * hour / 24))

    # Clinical context
    state[15] = float(ctx.get("concurrent_alerts", 0)) / 20.0
    state[16] = float(ctx.get("ward_occupancy", 25)) / 50.0

    return state.astype(np.float32)


def build_state_from_supabase_alert(alert_row: Dict, patient_row: Dict) -> np.ndarray:
    """
    Build state vector from raw Supabase alert_events and patients rows.
    Used by online_trainer when replaying historical experiences.
    """
    ac = alert_row.get("alert_context", {})
    if not ac:
        vs = alert_row.get("vitals_snapshot", {}) or {}
        patient_row = patient_row or {}
        ac = {
            "vitals": {
                "heart_rate": vs.get("heart_rate", alert_row.get("trigger_value", 100)),
                "spo2": vs.get("spo2", 95),
                "temperature": vs.get("temperature", 37.0),
                "sys_bp_est": vs.get("sys_bp_est", 120),
                "motion_score": vs.get("motion_score", 1),
            },
            "trajectory": {
                "hr_baseline_mean": patient_row.get("baseline_hr_mean", 75),
                "hr_baseline_std": patient_row.get("baseline_hr_std", 5),
                "spo2_baseline_mean": patient_row.get("baseline_spo2_mean", 97),
                "spo2_baseline_std": 1.0,
                "temp_baseline_mean": 36.8,
                "temp_baseline_std": 0.3,
                "bp_baseline_mean": 120,
                "bp_baseline_std": 8,
                "hr_first_derivative": 5.0,
                "hr_second_derivative": alert_row.get("second_derivative", 1.5),
                "spo2_second_derivative": 0.2,
            },
            "patient": {
                "age": _age_from_patient_row(patient_row),
                "genomic_risk": {
                    "cardiac": patient_row.get("genomic_risk_cardiac", "Low"),
                    "respiratory": patient_row.get("genomic_risk_respiratory", "Low"),
                    "diabetic": patient_row.get("genomic_risk_diabetic", "Low"),
                    "sepsis": patient_row.get("genomic_risk_sepsis", "Low"),
                },
            },
            "context": {"hour": 12, "concurrent_alerts": 0, "ward_occupancy": 20},
        }
    return build_state_vector(ac)
