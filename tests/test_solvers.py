"""Tests for solvers module."""

import numpy as np

import pytest

from solvers import FiniteVolumeSolver, GlacierParameters


@pytest.fixture
def finite_volume_solver() -> FiniteVolumeSolver:
    """Return instance of FiniteVolumeSolver."""
    initial_height = np.linspace(start=2, stop=0, num=11)
    x_coordinates = np.linspace(start=0, stop=10, num=11)
    return FiniteVolumeSolver(
        initial_height=initial_height, x_coordinates=x_coordinates
    )


def test_finite_volume_solver(finite_volume_solver):
    """Test plot method creating solution plot."""
    finite_volume_solver.plot(show=False)


def test_glacier_parameters():
    """GlacierParameters should contain relevant information."""
    initial_height = np.array([1, 2, 3, 2, 1, 0, 0, 0]).astype(float)
    x_coordinates = np.linspace(start=0, stop=700, num=8)
    accumulation_rate = np.array([2, 2, 1, 0, -1, 0, 0, 0]).astype(float)
    params = GlacierParameters(
        h_0=initial_height,
        xs=x_coordinates,
        alpha=np.radians(3),
        q=accumulation_rate,
    )
    assert params.g == 9.8
    assert params.rho == 917
    assert params.H == 3.0
    assert params.L == 400.0
    assert params.epsilon == 3.0 / 400.0
    assert params.alpha == np.radians(3)
    assert params.theta == 917 * 9.8 * 3 * np.sin(np.radians(3))
    assert params.Q == 2.0

    theta_cubed = (917 * 9.8 * 3 * np.sin(np.radians(3))) ** 3
    kappa = 2 * 3 * 1 * theta_cubed * (3 / 400) / 2
    assert params.kappa == kappa
    assert params.lambda_ == kappa / 5
