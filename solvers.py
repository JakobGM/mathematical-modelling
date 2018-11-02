"""Numerical solvers made specifically for glacier dynamics. """

import matplotlib.pyplot as plt

import numpy as np


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
