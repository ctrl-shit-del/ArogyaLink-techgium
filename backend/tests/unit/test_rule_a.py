"""Unit tests for Rule A (artifact rejection)."""
from datetime import datetime
from backend.core.trajectory.window_buffer import WindowBuffer, VitalReading
from backend.models.schemas.vitals import VitalPayload
from backend.core.rules.rule_a_artifact import rule_a_artifact


def test_rule_a_artifact_hr_spike():
    buffer = WindowBuffer(maxlen=10)
    buffer.append(VitalReading(heart_rate=80, spo2=97, temperature=37.0))
    payload = VitalPayload(patient_id="PT-001", heart_rate=125, spo2=97, temperature=37.0)  # +45 BPM
    assert rule_a_artifact(payload, buffer, hr_delta_max=40) is True


def test_rule_a_artifact_hr_normal():
    buffer = WindowBuffer(maxlen=10)
    buffer.append(VitalReading(heart_rate=80, spo2=97, temperature=37.0))
    payload = VitalPayload(patient_id="PT-001", heart_rate=95, spo2=97, temperature=37.0)  # +15 BPM
    assert rule_a_artifact(payload, buffer, hr_delta_max=40) is False


def test_rule_a_artifact_spo2_drop():
    buffer = WindowBuffer(maxlen=10)
    buffer.append(VitalReading(heart_rate=80, spo2=98, temperature=37.0))
    payload = VitalPayload(patient_id="PT-001", heart_rate=82, spo2=92, temperature=37.0)  # -6% SpO2
    assert rule_a_artifact(payload, buffer, spo2_delta_max=5) is True


def test_rule_a_artifact_first_reading():
    buffer = WindowBuffer(maxlen=10)
    payload = VitalPayload(patient_id="PT-001", heart_rate=200, spo2=80)
    assert rule_a_artifact(payload, buffer) is False  # no previous reading
