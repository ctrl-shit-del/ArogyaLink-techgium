"""Rule C: Trajectory — deviation>1.5σ AND accel>0 AND motion<2 → FIRE SYNERA_STATE."""
from backend.core.trajectory.calculator import get_trajectory_verdict
from backend.core.trajectory.window_buffer import WindowBuffer


def rule_c_trajectory(
    buffer: WindowBuffer,
    trigger_vital: str,
    baseline_mean: float,
    baseline_std: float,
    motion_score: int,
    sigma_threshold: float = 1.5,
    motion_max: int = 2,
    acceleration_window: int = 3,
) -> tuple[bool, dict]:
    """
    Returns (fires_synera: bool, verdict_dict).
    Fires when: deviation > sigma_threshold, sustained positive acceleration, motion <= motion_max.
    """
    verdict = get_trajectory_verdict(
        buffer=buffer,
        trigger_vital=trigger_vital,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        acceleration_window=acceleration_window,
    )
    if motion_score > motion_max:
        return False, verdict
    if verdict["deviation_sigma"] < sigma_threshold:
        return False, verdict
    if not verdict["is_sustained_acceleration"]:
        return False, verdict
    return True, verdict
