from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt

import numpy as np

from scipy import integrate


PhysicalVariable = namedtuple(
    'PhysicalVariable', field_names=('unscaled', 'scaled')
)


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
    mu: float = 9.3e-25

    def __post_init__(self) -> None:
        """Calculate derived constants."""
        # Approximated to be a small parameter
        self.epsilon: float = self.H / self.L

        # Maximum accumulation rate
        self.Q = np.abs(self.q).max()
        self.q = PhysicalVariable(unscaled=self.q, scaled=self.q / self.Q)

        # Scale other variables
        self.h_0 = PhysicalVariable(unscaled=self.h_0, scaled=self.h_0 / self.H)
        self.xs = PhysicalVariable(unscaled=self.xs, scaled=self.xs / self.L)

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
        xs = self.xs.scaled
        q = self.q.scaled
        integrated_q = integrate.cumtrapz(y=q, x=xs, initial=0)
        integral_constant = self.lambda_ * h_0 ** (self.m + 2)
        integrated_q += integral_constant
        integrated_q[integrated_q < 0.0] = 0.0
        return (integrated_q / self.lambda_) ** (1 / (self.m + 2))


class FiniteVolumeSolver:
    # A very naive CFL condition, not analytically found at all
    CFL: float = 0.1

    def __init__(self, glacier: GlacierParameters) -> None:
        self.glacier = glacier

    def solve(self, t_end: float, delta_t: Optional[float] = None) -> None:
        # Scale x coordinates
        xs = self.glacier.xs.scaled

        # Scale height coordinates
        h_0 = self.glacier.h_0.scaled

        # Spatial step used
        delta_x = xs[1] - xs[0]

        # Determine temporal time step
        delta_t = delta_t or self.CFL * delta_x

        num_t = int(t_end / delta_t)
        num_x = len(xs)

        h = np.zeros([num_t, num_x], dtype=float)
        h[:, 0] = h_0[0]
        h[0, :] = h_0

        lambda_ = self.glacier.lambda_
        q = self.glacier.q.scaled
        m = self.glacier.m

        from tqdm import tqdm

        for j in tqdm(np.arange(start=0, stop=num_t - 1)):
            flux_difference = lambda_ * (
                h[j, :-1] ** (m + 2) - h[j, 1:] ** (m + 2)
            )
            h[j + 1, 1:] = h[j, 1:] + (delta_t / delta_x) * (
                delta_x * q[1:] - flux_difference
            )
            np.nan_to_num(h, copy=False)
            h[j + 1, h[j + 1, :] < 0] = 0

        self.h = h * self.glacier.H

    def plot(self, show: bool = True) -> plt.Figure:
        """
        Plot solution and initial conditions.

        :param show: If True, the plot will be shown.
        :return: Matplotlib Figure object containing plot(s).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        ax1.set_title('Initial conditions')
        ax2.set_title('Final result')
        ax1.set_xlabel('$x$')
        ax2.set_xlabel('$x$')
        ax1.set_ylabel('$z$')

        ax1.fill(
            [0, *self.glacier.xs.unscaled], [0, *self.glacier.h_0.unscaled]
        )
        if hasattr(self, 'h'):
            ax2.fill([0, *self.glacier.xs], [0, *self.h[-1]])

        ax1.legend(['Glacier'])

        if show:
            plt.show()

        return fig


def simple_accumulation_model(
    snow_line: float,
    tongue: float,
    permanent_snow_rate: float,
    stop: float,
    num: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(start=0, stop=stop, num=num)
    slope = -2 * permanent_snow_rate * tongue / (tongue - snow_line) ** 2

    q = np.zeros(num)
    dx = stop / (num - 1)
    snow_line_index = int(snow_line * dx)
    q[: snow_line_index - 1] = permanent_snow_rate

    tongue_index = int(tongue * dx)
    slope_index_rate = slope * dx
    q[snow_line_index - 1 : tongue_index] = (
        slope_index_rate * np.arange(tongue_index - snow_line_index + 1)
        + permanent_snow_rate
    )
    return xs, q
