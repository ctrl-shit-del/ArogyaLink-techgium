"""Generates realistic vital trajectories for mock simulator."""
import math
import random
from datetime import datetime, timezone
from typing import List, Tuple


def flat_with_noise(baseline_hr: float, baseline_spo2: float, baseline_temp: float,
                    n_readings: int, noise_hr: float = 3, noise_spo2: float = 1,
                    noise_temp: float = 0.2) -> List[Tuple[float, float, float]]:
    """Gaussian noise around baseline. Returns list of (hr, spo2, temp)."""
    return [
        (
            max(40, baseline_hr + random.gauss(0, noise_hr)),
            max(85, min(100, baseline_spo2 + random.gauss(0, noise_spo2))),
            baseline_temp + random.gauss(0, noise_temp),
        )
        for _ in range(n_readings)
    ]


def exponential_acceleration(
    start_hr: float, end_hr: float, start_spo2: float, end_spo2: float,
    start_temp: float, end_temp: float, n_readings: int,
) -> List[Tuple[float, float, float]]:
    """Accelerating curve: HR rises with increasing rate, SpO2/temp follow."""
    out = []
    for i in range(n_readings):
        t = (i + 1) / n_readings
        # Exponential-like: faster rise toward end
        x = (math.exp(t * 2) - 1) / (math.e ** 2 - 1) if n_readings > 1 else 1.0
        hr = start_hr + (end_hr - start_hr) * x
        spo2 = start_spo2 + (end_spo2 - start_spo2) * x
        temp = start_temp + (end_temp - start_temp) * x
        out.append((hr, max(85, spo2), temp))
    return out


def exertion_spike(
    baseline_hr: float, baseline_spo2: float, peak_hr: float,
    rise_readings: int, total_readings: int, motion_score: int = 8,
) -> List[Tuple[float, float, float, int]]:
    """Fast rise to peak then return. Returns (hr, spo2, temp, motion)."""
    out = []
    for i in range(total_readings):
        if i < rise_readings:
            frac = (i + 1) / rise_readings
            hr = baseline_hr + (peak_hr - baseline_hr) * frac
        else:
            hr = baseline_hr + (peak_hr - baseline_hr) * 0.2 * (1 - (i - rise_readings) / max(1, total_readings - rise_readings))
        out.append((hr, baseline_spo2, 37.0, motion_score))
    return out


def glitch_spike(
    baseline_hr: float, baseline_spo2: float, baseline_temp: float,
    n_readings: int, glitch_value: float, glitch_index: int,
) -> List[Tuple[float, float, float]]:
    """One bad reading at glitch_index, rest normal."""
    out = []
    for i in range(n_readings):
        if i == glitch_index:
            out.append((glitch_value, baseline_spo2, baseline_temp))
        else:
            out.append((baseline_hr, baseline_spo2, baseline_temp))
    return out


def slow_linear_drift(
    start_hr: float, end_hr: float, baseline_spo2: float, baseline_temp: float,
    n_readings: int,
) -> List[Tuple[float, float, float]]:
    """Linear drift: constant rate, zero second derivative."""
    out = []
    for i in range(n_readings):
        t = i / max(1, n_readings - 1) if n_readings > 1 else 1.0
        hr = start_hr + (end_hr - start_hr) * t
        out.append((hr, baseline_spo2, baseline_temp))
    return out


def _trajectory_point(patient_id: str, profile: dict, reading_index: int, n: int = 60):
    """Return (hr, spo2, temp, motion) for the given reading index from the profile's trajectory."""
    traj = profile.get("trajectory", "flat_with_noise")
    bh = profile["baseline_hr"]
    bs = profile["baseline_spo2"]
    bt = profile.get("baseline_temp", 37.0)
    if traj == "flat_with_noise":
        pts = flat_with_noise(bh, bs, bt, n)
        pt = pts[reading_index % len(pts)]
        return (pt[0], pt[1], pt[2], 1)
    if traj == "exponential_acceleration":
        pts = exponential_acceleration(bh, 140, bs, 91, bt, 37.8, n)
        pt = pts[reading_index % len(pts)]
        return (pt[0], pt[1], pt[2], 1)
    if traj == "exertion_spike":
        pts = exertion_spike(bh, bs, 116, rise_readings=6, total_readings=n, motion_score=8)
        pt = pts[reading_index % len(pts)]
        return (pt[0], pt[1], pt[2], pt[3])
    if traj == "glitch_spike":
        pts = glitch_spike(bh, bs, bt, n, profile.get("glitch_value", 228), profile.get("glitch_index", 3))
        pt = pts[reading_index % len(pts)]
        return (pt[0], pt[1], pt[2], 1)
    if traj == "slow_linear_drift":
        pts = slow_linear_drift(bh, 92, bs, bt, n)
        pt = pts[reading_index % len(pts)]
        return (pt[0], pt[1], pt[2], 1)
    pts = flat_with_noise(bh, bs, bt, n)
    pt = pts[reading_index % len(pts)]
    return (pt[0], pt[1], pt[2], 1)


def generate_next_reading(patient_id: str, profile: dict, reading_index: int) -> dict:
    """Generate one vital reading for the given patient at the given index. HR for PT-0002 climbs 82→140 over trajectory."""
    hr, spo2, temp, motion = _trajectory_point(patient_id, profile, reading_index)
    return {
        "patient_id": patient_id,
        "vitals": {
            "heart_rate": round(hr, 1),
            "spo2": round(spo2, 1),
            "temperature": round(temp, 1),
        },
        "context": {"motion_score": motion},
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
