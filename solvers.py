from dataclasses import dataclass
from typing import Optional, Union

import matplotlib.pyplot as plt

import numpy as np

from scipy import integrate


@dataclass
class GlacierParameters:
    """Dataclass containing relevant information regarding a glacier."""

    # Initial height profile of glacier
    h_0: Union[float, np.ndarray]

    # And respective x-coordinates
    xs: np.ndarray

    # Accumulation rate of ice along the glacier
    q: np.ndarray

    # Slope of valley floor in radians
    alpha: float

    # Length scaling for glacier in meters
    L: float = 1000.0

    # Height scaling for glacier in meters
    H: float = 50.0

    # Gravitational acceleration in m/s^2
    g: float = 9.8

    # Density of the glacier ice
    rho: float = 917

    # Material constant from Glen's law, usually in range [1.8, 5]
    m: float = 3.0

    # Another material property from Glen's law
    # TODO: Find the typical value for this one!
    mu: float = 1.0

    def __post_init__(self) -> None:
        """Calculate derived constants."""
        # Approximated to be a small parameter
        self.epsilon: float = self.H / self.L

        # Maximum accumulation rate
        self.Q = self.q.max()

        # Stress scaler
        self.theta = self.rho * self.g * self.H * np.sin(self.alpha)

        # Derived constants used in differential equation
        self.kappa = (
            2
            * self.H
            * self.mu
            * (self.theta ** self.m)
            * self.epsilon
            / self.Q
        )
        self.lambda_ = self.kappa / (self.m + 2)

        if isinstance(self.h_0, (int, float)):
            self.h_0 = self.generate_steady_state_height(h_0=self.h_0)

    def generate_steady_state_height(self, h_0: float) -> np.ndarray:
        """Return height profile resulting in steady state, given q."""
        assert isinstance(h_0, (float, int))
        xs = self.xs / self.L
        integrated_q = integrate.cumtrapz(y=self.q, x=xs, initial=0)
        integral_constant = self.lambda_ * h_0 ** (self.m + 2)
        integrated_q += integral_constant
        integrated_q[integrated_q < 0.0] = 0.0
        return (integrated_q / self.lambda_) ** (1 / (self.m + 2))


class FiniteVolumeSolver:
    def __init__(self, glacier: GlacierParameters) -> None:
        self.initial_height = glacier.h_0
        self.x_coordinates = glacier.xs

    def solve(self) -> None:
        pass

    def plot(self, show: bool = True) -> plt.Figure:
        """
        Plot solution and initial conditions.

        :param show: If True, the plot will be shown.
        :return: Matplotlib Figure object containing plot(s).
        """
        fig, ax = plt.subplots()
        ax.set_title('Initial conditions')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$z$')

        ax.fill([0, *self.x_coordinates], [0, *self.initial_height])
        ax.legend(['Glacier'])

        if show:
            plt.show()

        return fig
