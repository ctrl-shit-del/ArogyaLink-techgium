"""Unit tests for Rule B (exertion)."""
from backend.core.trajectory.window_buffer import WindowBuffer, VitalReading
from backend.models.schemas.vitals import VitalPayload
from backend.core.rules.rule_b_exertion import rule_b_exertion


def test_rule_b_exertion_high_motion_hr_elevation():
    buffer = WindowBuffer(maxlen=10)
    buffer.append(VitalReading(heart_rate=68, motion_score=2))
    payload = VitalPayload(patient_id="PT-003", heart_rate=106, motion_score=8)
    assert rule_b_exertion(payload, buffer, baseline_hr=68, hr_elevation_threshold=15, motion_threshold=4) is True


def test_rule_b_exertion_low_motion():
    buffer = WindowBuffer(maxlen=10)
    payload = VitalPayload(patient_id="PT-002", heart_rate=119, motion_score=1)
    assert rule_b_exertion(payload, buffer, baseline_hr=82, hr_elevation_threshold=15, motion_threshold=4) is False


def test_rule_b_exertion_hr_not_high_enough():
    buffer = WindowBuffer(maxlen=10)
    payload = VitalPayload(patient_id="PT-001", heart_rate=85, motion_score=8)  # +10 from 75
    assert rule_b_exertion(payload, buffer, baseline_hr=75, hr_elevation_threshold=15, motion_threshold=4) is False
