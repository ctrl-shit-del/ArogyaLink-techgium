"""Calls derivatives and window buffer; returns trajectory verdict."""
from backend.core.trajectory.derivatives import (
    compute_first_derivative,
    compute_second_derivative,
    is_sustained_acceleration,
    compute_sigma_deviation,
)
from backend.core.trajectory.window_buffer import WindowBuffer, VitalReading

# Default 5-second interval between readings
DT = 5.0


def get_trajectory_verdict(
    buffer: WindowBuffer,
    trigger_vital: str,
    baseline_mean: float,
    baseline_std: float,
    acceleration_window: int = 3,
) -> dict:
    """
    Returns verdict dict with:
    - deviation_sigma: float
    - second_derivative: float (at latest reading)
    - is_sustained_acceleration: bool
    - current_value: float
    - rates: list[float] (first derivative)
    - accelerations: list[float] (second derivative)
    """
    if trigger_vital == "heart_rate":
        values = buffer.get_heart_rates()
    elif trigger_vital == "spo2":
        values = buffer.get_spo2()
    elif trigger_vital == "temperature":
        values = buffer.get_temperatures()
    else:
        values = buffer.get_heart_rates()  # fallback

    if len(values) < 3:
        return {
            "deviation_sigma": 0.0,
            "second_derivative": 0.0,
            "is_sustained_acceleration": False,
            "current_value": values[-1] if values else 0.0,
            "rates": [],
            "accelerations": [],
        }

    current = values[-1]
    deviation_sigma = compute_sigma_deviation(current, baseline_mean, baseline_std)
    rates = compute_first_derivative(values, DT)
    accelerations = compute_second_derivative(rates, DT)
    second_deriv = accelerations[-1] if accelerations else 0.0
    sustained = is_sustained_acceleration(accelerations, window=acceleration_window)

    # Fallback: also treat as sustained acceleration if the last `window` first-derivatives
    # are all positive AND the sigma deviation is ≥ 1.5  (catches the steep-plateau case
    # where the 10-reading buffer sees only a high constant rate, making 2nd deriv ≈ 0).
    if not sustained and deviation_sigma >= 1.5 and len(rates) >= acceleration_window:
        tail_rates = rates[-acceleration_window:]
        if sum(1 for r in tail_rates if r > 0) >= acceleration_window:
            sustained = True

    return {
        "deviation_sigma": deviation_sigma,
        "second_derivative": second_deriv,
        "is_sustained_acceleration": sustained,
        "current_value": current,
        "rates": rates,
        "accelerations": accelerations,
    }
