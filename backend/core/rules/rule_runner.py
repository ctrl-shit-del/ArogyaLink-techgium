"""Runs rules A → B → C in sequence. Returns RuleResult."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from backend.models.schemas.vitals import VitalPayload
from backend.core.trajectory.window_buffer import WindowBuffer, VitalReading
from backend.core.rules.rule_a_artifact import rule_a_artifact
from backend.core.rules.rule_b_exertion import rule_b_exertion
from backend.core.rules.rule_c_trajectory import rule_c_trajectory


class RuleResult(Enum):
    ARTIFACT = "ARTIFACT"       # Rule A rejected — discard packet
    EXERTION = "EXERTION"       # Rule B — log as routine exertion
    SYNERA_STATE = "SYNERA_STATE"  # Rule C — fire alert + trigger RAG
    STABLE = "STABLE"           # All rules passed, no alert condition
    WATCH = "WATCH"             # Deviation >1σ but no acceleration yet


@dataclass
class RuleRunnerConfig:
    artifact_hr_delta: int = 40
    artifact_spo2_delta: int = 5
    artifact_temp_delta: float = 0.5
    exertion_motion_threshold: int = 4
    exertion_hr_elevation: int = 15
    synera_sigma_threshold: float = 1.5
    synera_motion_max: int = 2
    synera_acceleration_window: int = 3


def _payload_to_reading(payload: VitalPayload) -> VitalReading:
    return VitalReading(
        heart_rate=payload.heart_rate,
        spo2=payload.spo2,
        temperature=payload.temperature,
        sys_bp_est=payload.sys_bp_est,
        dia_bp_est=payload.dia_bp_est,
        motion_score=payload.motion_score,
        recorded_at=payload.recorded_at.timestamp() if payload.recorded_at else None,
    )


def run(
    payload: VitalPayload,
    buffer: WindowBuffer,
    baseline_hr_mean: float,
    baseline_hr_std: float,
    baseline_spo2_mean: Optional[float] = None,
    baseline_spo2_std: Optional[float] = None,
    baseline_temp_mean: Optional[float] = None,
    baseline_temp_std: Optional[float] = None,
    config: Optional[RuleRunnerConfig] = None,
) -> tuple[RuleResult, dict]:
    """
    Sequential pipeline A → B → C.
    Returns (RuleResult, extra_dict with verdict/trigger_vital/etc for SYNERA_STATE/WATCH).
    """
    cfg = config or RuleRunnerConfig()

    # Rule A: Artifact
    if rule_a_artifact(
        payload,
        buffer,
        hr_delta_max=cfg.artifact_hr_delta,
        spo2_delta_max=cfg.artifact_spo2_delta,
        temp_delta_max=cfg.artifact_temp_delta,
    ):
        return RuleResult.ARTIFACT, {}

    # Append to buffer for trajectory calc (we're past artifact check)
    buffer.append(_payload_to_reading(payload))
    motion = payload.motion_score if payload.motion_score is not None else 0

    # Rule B: Exertion
    if rule_b_exertion(
        payload,
        buffer,
        baseline_hr_mean,
        hr_elevation_threshold=cfg.exertion_hr_elevation,
        motion_threshold=cfg.exertion_motion_threshold,
    ):
        return RuleResult.EXERTION, {"motion_score": motion, "hr_elevation": (payload.heart_rate or 0) - baseline_hr_mean}

    # Determine primary trigger vital (simplified: use HR if deviation is highest; else spo2/temp)
    trigger_vital = "heart_rate"
    baseline_mean = baseline_hr_mean
    baseline_std = baseline_hr_std or 1.0
    if baseline_spo2_mean is not None and payload.spo2 is not None and baseline_spo2_std:
        from backend.core.trajectory.derivatives import compute_sigma_deviation
        hr_sigma = compute_sigma_deviation(payload.heart_rate or 0, baseline_hr_mean, baseline_hr_std or 1.0)
        spo2_sigma = compute_sigma_deviation(payload.spo2, baseline_spo2_mean, baseline_spo2_std)
        if spo2_sigma > hr_sigma:
            trigger_vital = "spo2"
            baseline_mean = baseline_spo2_mean
            baseline_std = baseline_spo2_std
    if trigger_vital == "heart_rate" and baseline_temp_mean is not None and payload.temperature is not None and baseline_temp_std:
        from backend.core.trajectory.derivatives import compute_sigma_deviation
        temp_sigma = compute_sigma_deviation(payload.temperature, baseline_temp_mean, baseline_temp_std)
        hr_sigma = compute_sigma_deviation(payload.heart_rate or 0, baseline_hr_mean, baseline_hr_std or 1.0)
        if temp_sigma > hr_sigma:
            trigger_vital = "temperature"
            baseline_mean = baseline_temp_mean
            baseline_std = baseline_temp_std

    # Rule C: Trajectory
    fires, verdict = rule_c_trajectory(
        buffer,
        trigger_vital=trigger_vital,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        motion_score=motion,
        sigma_threshold=cfg.synera_sigma_threshold,
        motion_max=cfg.synera_motion_max,
        acceleration_window=cfg.synera_acceleration_window,
    )
    if fires:
        return RuleResult.SYNERA_STATE, {
            "trigger_vital": trigger_vital,
            "trigger_value": verdict["current_value"],
            "baseline_value": baseline_mean,
            "deviation_sigma": verdict["deviation_sigma"],
            "second_derivative": verdict["second_derivative"],
            "motion_score": motion,
            "verdict": verdict,
        }

    # Deviation > 1σ but no acceleration → WATCH
    if verdict.get("deviation_sigma", 0) >= 1.0:
        return RuleResult.WATCH, {
            "trigger_vital": trigger_vital,
            "deviation_sigma": verdict["deviation_sigma"],
            "verdict": verdict,
        }

    return RuleResult.STABLE, {}
