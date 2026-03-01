"""Rule B: Exertion — HR elev >15 BPM AND motion_score >4 → exertion (log only, no alert)."""
from backend.models.schemas.vitals import VitalPayload
from backend.core.trajectory.window_buffer import WindowBuffer


def rule_b_exertion(
    payload: VitalPayload,
    buffer: WindowBuffer,
    baseline_hr: float,
    hr_elevation_threshold: int = 15,
    motion_threshold: int = 4,
) -> bool:
    """
    Returns True if this is exertion (high motion + HR elevation).
    When True: log as routine exertion, do not fire alert.
    """
    motion = payload.motion_score if payload.motion_score is not None else 0
    if motion <= motion_threshold:
        return False
    hr = payload.heart_rate
    if hr is None or baseline_hr is None:
        return False
    return (hr - baseline_hr) > hr_elevation_threshold
