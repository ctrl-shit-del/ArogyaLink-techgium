"""Unit tests for trajectory derivatives."""
import pytest
from backend.core.trajectory.derivatives import (
    compute_first_derivative,
    compute_second_derivative,
    is_sustained_acceleration,
    compute_sigma_deviation,
)


def test_compute_first_derivative_central():
    # Linear: 0, 5, 10, 15, 20 (dt=5) -> rate = 1.0
    readings = [0.0, 5.0, 10.0, 15.0, 20.0]
    rates = compute_first_derivative(readings, dt=5.0)
    assert len(rates) == 5
    assert abs(rates[2] - 1.0) < 0.01  # central at index 2
    assert abs(rates[3] - 1.0) < 0.01


def test_compute_second_derivative_constant_rate():
    rates = [1.0, 1.0, 1.0, 1.0]
    acc = compute_second_derivative(rates, dt=5.0)
    assert len(acc) == 4
    # Constant rate -> zero acceleration in middle
    assert abs(acc[1]) < 0.01
    assert abs(acc[2]) < 0.01


def test_is_sustained_acceleration_positive_increasing():
    # Last 3 positive and strictly increasing
    acc = [0.1, 0.2, 0.3]
    assert is_sustained_acceleration(acc, window=3) is True


def test_is_sustained_acceleration_single_spike():
    acc = [0.0, 0.0, 1.0]  # single spike at end
    assert is_sustained_acceleration(acc, window=3) is False  # not all positive + increasing (first two 0)


def test_is_sustained_acceleration_flat_positive():
    acc = [0.5, 0.5, 0.5]  # positive but not increasing
    assert is_sustained_acceleration(acc, window=3) is False


def test_compute_sigma_deviation():
    assert abs(compute_sigma_deviation(110, 100, 10) - 1.0) < 0.01
    assert compute_sigma_deviation(100, 100, 0) == 0.0
