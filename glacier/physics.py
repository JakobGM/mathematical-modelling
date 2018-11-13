import pickle
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Union

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
            # assert self.q_0 and self.x_s and self.x_f
            self.create_simple_accumulation_model()
        self.q = PhysicalVariable(unscaled=self.q, scaled=self.q / self.Q)

        if isinstance(self.h_0, (int, float)):
            self.h_0 = self.generate_steady_state_height()
        self.h_0 = PhysicalVariable(unscaled=self.h_0, scaled=self.h_0 / self.H)

    def generate_steady_state_height(self) -> np.ndarray:
        """Return height profile resulting in steady state, given q."""
        if isinstance(self.h_0, PhysicalVariable):
            h_0 = self.h_0.scaled[0]
        else:
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
        q = self.q.unscaled.copy() * 3600 * 24 * 365
        print(q)
        # tail_length = len(hs) - len(np.trim_zeros(hs, trim='b'))
        # q[-tail_length:] = 0

        ax2.plot(xs, q * (3600 * 24 * 365), color='tab:red', alpha=0.7)
        ax2.set_ylabel('$q$')
        ax2.legend(['Accumulation rate'], loc='lower right')

        if show:
            plt.show()

        return fig
