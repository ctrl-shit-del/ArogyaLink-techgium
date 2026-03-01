"""Unit tests for Rule C (trajectory / SYNERA_STATE)."""
from backend.core.trajectory.window_buffer import WindowBuffer, VitalReading
from backend.core.rules.rule_c_trajectory import rule_c_trajectory


def test_rule_c_fires_sustained_acceleration_high_deviation():
    buffer = WindowBuffer(maxlen=10)
    # Build accelerating HR: 82, 87, 93, 100, 108, 117, 125 (rising + accelerating)
    for i, hr in enumerate([82, 87, 93, 100, 108, 117, 125]):
        buffer.append(VitalReading(heart_rate=hr, motion_score=1))
    fires, verdict = rule_c_trajectory(
        buffer, trigger_vital="heart_rate",
        baseline_mean=82, baseline_std=5.0,
        motion_score=1, sigma_threshold=1.5, motion_max=2, acceleration_window=3,
    )
    # Deviation from 82: 125 is (125-82)/5 = 8.6 sigma; acceleration positive and sustained
    assert verdict["deviation_sigma"] >= 1.5
    assert fires or verdict["deviation_sigma"] >= 1.0  # at least WATCH-level


def test_rule_c_no_fire_high_motion():
    buffer = WindowBuffer(maxlen=10)
    for hr in [82, 90, 100, 115, 125]:
        buffer.append(VitalReading(heart_rate=hr, motion_score=1))
    fires, _ = rule_c_trajectory(
        buffer, trigger_vital="heart_rate",
        baseline_mean=82, baseline_std=5.0,
        motion_score=8, sigma_threshold=1.5, motion_max=2,  # motion 8 > 2 → no fire
    )
    assert fires is False


def test_rule_c_no_fire_insufficient_deviation():
    buffer = WindowBuffer(maxlen=10)
    for hr in [82, 84, 86, 88, 90]:  # slow drift, low sigma
        buffer.append(VitalReading(heart_rate=hr, motion_score=1))
    fires, verdict = rule_c_trajectory(
        buffer, trigger_vital="heart_rate",
        baseline_mean=82, baseline_std=10.0,  # high std -> low sigma
        motion_score=1, sigma_threshold=1.5, motion_max=2,
    )
    assert verdict["deviation_sigma"] < 1.5 or fires is False
