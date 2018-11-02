"""Numerical solvers made specifically for glacier dynamics. """

from dataclasses import dataclass

import matplotlib.pyplot as plt

import numpy as np


@dataclass
class GlacierParameters:
    """Dataclass containing relevant information regarding a glacier."""

    # Initial height profile of glacier
    h_0: np.ndarray

    # And respective x-coordinates
    xs: np.ndarray

    # Accumulation rate of ice along the glacier
    q: np.ndarray

    # Slope of valley floor in radians
    alpha: float

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
        # Max height of glacier
        self.H: float = self.h_0.max()

        # Length of glacier
        # Assuming h > 0 for all x < x_F
        self.L: float = self.xs[self.h_0 > 0][-1]

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


class FiniteVolumeSolver:
    def __init__(
        self, initial_height: np.ndarray, x_coordinates: np.ndarray
    ) -> None:
        self.initial_height = initial_height
        self.x_coordinates = x_coordinates

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
