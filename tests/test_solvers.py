"""Tests for solvers module."""

import numpy as np

import pytest

from solvers import FiniteVolumeSolver


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
