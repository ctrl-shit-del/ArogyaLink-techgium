"""3-rule sequential pipeline runner — delegates to rule_runner.run()."""
from backend.models.schemas.vitals import VitalPayload
from backend.core.trajectory.window_buffer import WindowBuffer
from backend.core.rules.rule_runner import run as rule_run, RuleResult, RuleRunnerConfig


def run_pipeline(
    payload: VitalPayload,
    buffer: WindowBuffer,
    baseline_hr_mean: float,
    baseline_hr_std: float = 0.0,
    baseline_spo2_mean: float | None = None,
    baseline_spo2_std: float | None = None,
    baseline_temp_mean: float | None = None,
    baseline_temp_std: float | None = None,
    config: RuleRunnerConfig | None = None,
) -> tuple[RuleResult, dict]:
    """Runs A → B → C via rule_runner.run. Returns (RuleResult, extra_dict)."""
    return rule_run(
        payload=payload,
        buffer=buffer,
        baseline_hr_mean=baseline_hr_mean,
        baseline_hr_std=baseline_hr_std or 1.0,
        baseline_spo2_mean=baseline_spo2_mean,
        baseline_spo2_std=baseline_spo2_std,
        baseline_temp_mean=baseline_temp_mean,
        baseline_temp_std=baseline_temp_std,
        config=config,
    )
