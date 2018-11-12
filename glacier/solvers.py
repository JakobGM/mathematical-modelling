import pickle
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np

from glacier.flow_field import stationary_internal_flow_field
from glacier.physics import GlacierParameters


class Solver:
    CFL: float

    def __init__(self, glacier: GlacierParameters) -> None:
        self.glacier = glacier

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
            ax2.fill([0, *self.glacier.xs.unscaled], [0, *self.h[-1]])

        ax1.legend(['Glacier'])

        if show:
            plt.show()

        return fig

    def save(self, name: str) -> None:
        with open(self.get_filepath(name), 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, name: str) -> 'Solver':
        with open(cls.get_filepath(name), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_filepath(name: str) -> Path:
        directory = Path(__file__).parents[1] / 'results'
        directory.mkdir(parents=False, exist_ok=True)
        return directory / (name + '_solver.pickle')

    def calculate_flow_fields(self, step: int) -> None:
        if hasattr(self, 'Us'):
            return

        xs = self.glacier.xs.unscaled
        angle = np.degrees(self.glacier.alpha)
        q = [self.glacier.q.unscaled]

        U_scale = self.glacier.Q * self.glacier.L / self.glacier.H
        V_scale = self.glacier.Q
        z_scale = self.glacier.H

        self.flow_field_step = step
        self.Us = []
        self.Vs = []
        self.zs = []
        for height in tqdm(self.h[::step]):
            U, V, _, z = stationary_internal_flow_field(
                xs=xs, h_0=height, angle=angle, production=q
            )
            self.Us.append(U_scale * U)
            self.Vs.append(V_scale * V)
            self.zs.append(z_scale * z)

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

        # Assigning to self right away to prevent MemoryError at later
        # assignment
        self.h = np.zeros([num_t, num_x], dtype=float)
        h = self.h
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

        self.h *= self.glacier.H
