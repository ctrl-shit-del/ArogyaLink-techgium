"""Integration: pipeline end-to-end with mock payloads."""
import pytest
from backend.core.trajectory.window_buffer import WindowBuffer, VitalReading
from backend.models.schemas.vitals import VitalPayload
from backend.core.rules.rule_runner import run, RuleResult, RuleRunnerConfig


def test_pipeline_artifact_discard():
    buffer = WindowBuffer(maxlen=10)
    buffer.append(VitalReading(heart_rate=80, spo2=97))
    payload = VitalPayload(patient_id="PT-004", heart_rate=228, spo2=97)  # glitch spike
    result, _ = run(payload, buffer, baseline_hr_mean=79, baseline_hr_std=5.0, config=RuleRunnerConfig())
    assert result == RuleResult.ARTIFACT


def test_pipeline_exertion_logged():
    buffer = WindowBuffer(maxlen=10)
    for _ in range(3):
        buffer.append(VitalReading(heart_rate=68, motion_score=2))
    payload = VitalPayload(patient_id="PT-003", heart_rate=106, motion_score=8)
    result, extra = run(payload, buffer, baseline_hr_mean=68, baseline_hr_std=4.0, config=RuleRunnerConfig())
    assert result == RuleResult.EXERTION
