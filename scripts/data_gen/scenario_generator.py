"""Generates realistic vital trajectories for mock simulator."""
import math
import random
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
