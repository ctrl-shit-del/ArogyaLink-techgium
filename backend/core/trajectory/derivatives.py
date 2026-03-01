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
    """True if last `window` second derivatives are ALL positive AND increasing.
    Single-point spikes must NOT trigger — must be sustained."""
    n = len(second_derivatives)
    if n < window:
        return False
    tail = second_derivatives[-window:]
    if any(x <= 0 for x in tail):
        return False
    # Check strictly increasing
    for i in range(1, len(tail)):
        if tail[i] <= tail[i - 1]:
            return False
    return True


def compute_sigma_deviation(current: float, mean: float, std: float) -> float:
    """How many standard deviations current reading is from personal baseline."""
    if std == 0:
        return 0.0
    return abs(current - mean) / std
