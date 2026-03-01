"""Rule A: Artifact rejection — ±40 BPM delta, ±5% SpO₂, ±0.5°C temp. Reject packet if spike."""
from typing import Optional

from backend.models.schemas.vitals import VitalPayload
from backend.core.trajectory.window_buffer import WindowBuffer, VitalReading


def rule_a_artifact(
    payload: VitalPayload,
    buffer: WindowBuffer,
    hr_delta_max: int = 40,
    spo2_delta_max: int = 5,
    temp_delta_max: float = 0.5,
) -> bool:
    """
    Returns True if packet is ARTIFACT (reject/discard).
    Checks: HR jump > hr_delta_max, SpO2 jump > spo2_delta_max, temp jump > temp_delta_max.
    """
    last = buffer.last_reading()
    if last is None:
        return False  # First reading, allow

    if payload.heart_rate is not None and last.heart_rate is not None:
        if abs(payload.heart_rate - last.heart_rate) >= hr_delta_max:
            return True
    if payload.spo2 is not None and last.spo2 is not None:
        if abs(payload.spo2 - last.spo2) >= spo2_delta_max:
            return True
    if payload.temperature is not None and last.temperature is not None:
        if abs(payload.temperature - last.temperature) >= temp_delta_max:
            return True
    return False
