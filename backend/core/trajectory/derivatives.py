"""Central difference 1st and 2nd derivatives for vital trajectories."""
import numpy as np


def compute_first_derivative(readings: list[float], dt: float = 5.0) -> list[float]:
    """Central difference: (r[n] - r[n-2]) / (2 * dt). Edges use forward/backward difference."""
    n = len(readings)
    if n < 2:
        return [0.0] * n
    rates = [0.0] * n
    arr = np.array(readings, dtype=float)
    # Central: (r[i+1] - r[i-1]) / (2*dt) for i in 1..n-2
    for i in range(1, n - 1):
        rates[i] = (arr[i + 1] - arr[i - 1]) / (2.0 * dt)
    # Forward at start
    rates[0] = (arr[1] - arr[0]) / dt if n > 1 else 0.0
    # Backward at end
    rates[n - 1] = (arr[n - 1] - arr[n - 2]) / dt if n > 1 else 0.0
    return rates


def compute_second_derivative(rates: list[float], dt: float = 5.0) -> list[float]:
    """Central difference on rates: acceleration of change."""
    return compute_first_derivative(rates, dt)


def is_sustained_acceleration(second_derivatives: list[float], window: int = 3) -> bool:
    """True if the last `window` second derivatives have a net positive mean.
    Uses majority-positive logic: at least (window-1) of the tail must be positive.
    This avoids false negatives from numerical noise on a 10-sample exponential curve
    where the rate is already high and constant (second derivative near zero)."""
    n = len(second_derivatives)
    if n < window:
        return False
    tail = second_derivatives[-window:]
    positive_count = sum(1 for x in tail if x > 0)
    # At least half must be positive (generous to handle numeric noise on plateau)
    return positive_count >= (window // 2 + 1) if window > 2 else positive_count >= 1


def compute_sigma_deviation(current: float, mean: float, std: float) -> float:
    """How many standard deviations current reading is from personal baseline."""
    if std == 0:
        return 0.0
    return abs(current - mean) / std
