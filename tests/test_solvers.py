"""Tests for solvers module."""

import numpy as np

import pytest

from solvers import (
    FiniteVolumeSolver,
    GlacierParameters,
    simple_accumulation_model,
)


@pytest.fixture
def finite_volume_solver() -> FiniteVolumeSolver:
    """Return instance of FiniteVolumeSolver."""
    initial_height = np.linspace(start=2, stop=0, num=11)
    x_coordinates = np.linspace(start=0, stop=10, num=11)
    accumulation = np.array([2, 2, 2, 2, 1, 0, -1, -2, 0, 0, 0])
    glacier = GlacierParameters(
        h_0=initial_height,
        xs=x_coordinates,
        q=accumulation,
        alpha=np.radians(3),
    )
    return FiniteVolumeSolver(glacier)


def test_simple_accumulation_model():
    _, q = simple_accumulation_model(
        snow_line=3, tongue=9, permanent_snow_rate=2, num=11, stop=10
    )
    np.testing.assert_array_equal(
        q, np.array([2, 2, 2, 1, 0, -1, -2, -3, -4, 0, 0])
    )


def test_plotting_initial_conditions():
    xs, q = simple_accumulation_model(
        snow_line=300,
        tongue=600,
        stop=1000,
        permanent_snow_rate=1,
        num=1001,
        h_0=50,
    )
    glacier = GlacierParameters(h_0=50, xs=xs, q=q, alpha=np.radians(3))
    glacier.plot(show=False)


def test_finite_volume_solver(finite_volume_solver):
    """Test plot method creating solution plot."""
    finite_volume_solver.plot(show=False)


@pytest.mark.skip('')
def test_solving_with_finite_volume_method(finite_volume_solver):
    num = 3
    glacier = GlacierParameters(
        h_0=np.linspace(start=1, stop=0, num=num),
        xs=np.linspace(start=0, stop=1, num=num),
        q=np.linspace(start=0, stop=0.0000000000000000000001, num=num),
        alpha=np.radians(3),
    )
    finite_volume_solver = FiniteVolumeSolver(glacier)
    finite_volume_solver.solve(t_end=1.0, delta_t=0.0001)
    # finite_volume_solver.plot()


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
        H=3.0,
        L=400.0,
        mu=1,
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


def test_generate_steady_state_height_of_glacier():
    xs = np.array([0, 1, 2, 3, 4, 5])
    accumulation_rate = np.array([1, 0, -1, -2, -3, -4])
    glacier = GlacierParameters(
        xs=xs, q=accumulation_rate, h_0=2, alpha=np.radians(3)
    )
    assert glacier


def test_lol():
    x_max = 1000
    n = 100
    xs = np.linspace(0, x_max, n)
    q = np.linspace(1, -2, n)
    gs = GlacierParameters(xs=xs, q=q, h_0=0.0003, alpha=np.radians(5))
    import matplotlib.pyplot as plt

    plt.plot(gs.xs, gs.h_0)
    assert not np.any(np.isnan(gs.h_0))
    print(gs.h_0)
