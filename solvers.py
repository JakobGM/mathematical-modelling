from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Union
from tqdm import tqdm

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

    # Slope of valley floor in radians
    alpha: float

    # Accumulation rate of ice along the glacier
    q: Optional[np.ndarray] = None

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
    mu: float = 9.3e-21

    # Simple accumulation model
    q_0: Optional[float] = None
    x_s: Optional[float] = None
    x_f: Optional[float] = None

    def __post_init__(self) -> None:
        """Calculate derived constants."""
        # Approximated to be a small parameter
        seconds_in_year = 3600 * 24 * 365
        if self.q is not None:
            self.q = self.q / seconds_in_year
            assert self.q_0 is None
        if self.q_0 is not None:
            self.q_0 = self.q_0 / seconds_in_year
            assert self.q is None

        self.epsilon: float = self.H / self.L

        # Scale other variables
        self.xs = PhysicalVariable(unscaled=self.xs, scaled=self.xs / self.L)

        if self.q is not None:
            self.Q = np.abs(self.q).max() or 1 / seconds_in_year
        else:
            assert self.q_0
            self.Q = self.q_0

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

        if self.q is None:
            assert self.q_0 and self.x_s and self.x_f
            self.create_simple_accumulation_model()
        self.q = PhysicalVariable(unscaled=self.q, scaled=self.q / self.Q)

        if isinstance(self.h_0, (int, float)):
            self.h_0 = self.generate_steady_state_height(h_0=self.h_0)
        self.h_0 = PhysicalVariable(unscaled=self.h_0, scaled=self.h_0 / self.H)

    def generate_steady_state_height(self, h_0: float) -> np.ndarray:
        """Return height profile resulting in steady state, given q."""
        assert isinstance(h_0, (float, int))
        h_0 = self.h_0 / self.H
        xs = self.xs.scaled
        q = self.q.scaled
        integrated_q = integrate.cumtrapz(y=q, x=xs, initial=0) / self.lambda_
        integrated_q += h_0 ** (self.m + 2)
        integrated_q[integrated_q < 0.0] = 0.0
        return integrated_q ** (1 / (self.m + 2)) * self.H

    def create_simple_accumulation_model(self):
        xs = self.xs.scaled

        self.q_0 = PhysicalVariable(unscaled=self.q_0, scaled=self.q_0 / self.Q)
        q_0 = self.q_0.scaled

        self.x_f = PhysicalVariable(unscaled=self.x_f, scaled=self.x_f / self.L)
        x_f = self.x_f.scaled

        self.x_s = PhysicalVariable(unscaled=self.x_s, scaled=self.x_s / self.L)
        x_s = self.x_s.scaled

        if isinstance(self.h_0, np.ndarray):
            h_0 = self.h_0[0] / self.H
        elif isinstance(self.h_0, (int, float)):
            h_0 = self.h_0 / self.H
        else:
            raise ValueError

        slope = (
            -2
            * (q_0 * x_f + self.lambda_ * h_0 ** (self.m + 2))
            / (x_f - x_s) ** 2
        )

        num = len(xs)
        stop = xs[-1]
        q = np.zeros(num)
        dx = stop / (num - 1)
        snow_line_index = int(x_s / dx)
        q[:snow_line_index] = q_0

        tongue_index = int(x_f / dx)
        slope_index_rate = slope * dx
        q[snow_line_index:] = (
            slope_index_rate * np.arange(num - snow_line_index) + q_0
        )
        self.q = self.Q * q

    def plot(self, show: bool = True) -> plt.Figure:
        fig, ax = plt.subplots(1, 1)
        ax.set_title('Initial conditions')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$z$')
        xs = self.xs.unscaled
        hs = self.h_0.unscaled

        ax.fill([0, *xs, xs[-1]], [0, *hs, 0], alpha=0.7)
        ax.legend(['Glacier'])
        ax.set_xlim(0, xs[-1])

        ax2 = ax.twinx()

        # Set zero production from glacier toe and forwards
        q = self.q.unscaled.copy()
        tail_length = len(hs) - len(np.trim_zeros(hs, trim='b'))
        q[-tail_length:] = 0

        ax2.plot(xs, q * (3600 * 24 * 365), color='tab:red', alpha=0.7)
        ax2.set_ylabel('$q$')
        ax2.legend(['Accumulation rate'], loc='lower right')

        if show:
            plt.show()

        return fig


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

        lambda_ = self.glacier.lambda_
        m = self.glacier.m

        # Determine temporal time step
        delta_t = delta_t or 0.1 * 0.5 * delta_x / lambda_  # naive CFL
        # delta_t = delta_t or delta_x / (kappa * 2**(m+1)) # less naive?

        num_t = int(t_end / delta_t)
        num_x = len(xs)

        h = np.zeros([num_t, num_x], dtype=float)
        h[:, 0] = h_0[0]
        h[0, :] = h_0

        q = self.glacier.q.scaled

        q_trapez = (delta_t / 2) * (q[:-1] + q[1:])
        C = lambda_ * delta_t / delta_x

        for j in tqdm(np.arange(start=0, stop=num_t - 1)):
            now = h[j]
            future = h[j + 1]
            flux_difference = now[1:] ** int(m + 2) - now[:1] ** int(m + 2)
            future[1:] = now[1:] + q_trapez + C * flux_difference

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


class UpwindSolver:
    def __init__(self, glacier: GlacierParameters) -> None:
        self.glacier = glacier

    def solve(
        self, t_end: float, delta_t: Optional[float] = None, method=1
    ) -> None:
        # Scale x coordinates
        xs = self.glacier.xs.scaled

        # Scale height coordinates
        h_0 = self.glacier.h_0.scaled

        # Spatial step used
        delta_x = xs[1] - xs[0]

        lambda_ = self.glacier.lambda_
        kappa = self.glacier.kappa
        m = self.glacier.m

        # Determine time step
        # TODO: Find suitable time step (check if stable for larger
        # TODO: step, check if the two methods can use different steps
        # delta_t = delta_t or 0.5 * delta_x / lambda_  # naive CFL
        delta_t = delta_t or 2 * delta_x / (kappa * 2 ** (m + 1))  # less naive?

        num_t = int(t_end / delta_t)
        num_x = len(xs)

        h = np.zeros([num_t, num_x], dtype=float)
        h[:, 0] = h_0[0]
        h[0, :] = h_0

        q = self.glacier.q.scaled
        q_negative_indices = q < 0

        # Constant used in numerical scheme
        if method == "upwind":
            C1 = kappa * delta_t / delta_x
        elif method == "finite volume":
            C1 = lambda_ * delta_t / delta_x

        for j in tqdm(np.arange(start=0, stop=num_t - 1)):
            # No melting where there is no ice
            no_ice_indices = h[j, :] == 0
            this_q = q.copy()
            this_q[np.logical_and(no_ice_indices, q_negative_indices)] = 0

            if method == "upwind":
                h[j + 1, 1:] = (
                    h[j, 1:]
                    + (
                        this_q[1:] * delta_t
                        - C1 * h[j, 1:] ** (m + 1) * (h[j, 1:] - h[j, :-1])
                    )
                ).clip(min=0)
            elif method == "finite volume":
                h[j + 1, 1:] = (
                    h[j, 1:]
                    + (
                        this_q[1:] * delta_t
                        - C1 * (h[j, 1:] ** (m + 2) - h[j, :-1] ** (m + 2))
                    )
                ).clip(min=0)

            assert not np.isnan(np.sum(h[j + 1, 1:]))
            assert np.all(h[j + 1, 1:] >= 0)

        self.h = h * self.glacier.H
