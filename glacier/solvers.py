import abc
import pickle
from typing import Optional
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np

from glacier.flow_field import stationary_internal_flow_field
from glacier.physics import GlacierParameters


class Solver(abc.ABC):
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

    @abc.abstractmethod
    def solve(self, t_end: float, delta_t: Optional[float] = None) -> None:
        raise NotImplementedError

    def save(self, name: str) -> None:
        with open(name + '_solver.pickle', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(name: str) -> 'Solver':
        with open(name + '_solver.pickle', 'rb') as f:
            return pickle.load(f)

    def calculate_flow_fields(self, step: int) -> None:
        if hasattr(self, 'Us'):
            return

        xs = self.glacier.xs.unscaled
        angle = np.degrees(self.glacier.alpha)
        q = [self.glacier.q.unscaled]

        self.flow_field_step = step
        self.Us = []
        self.Vs = []
        self.zs = []
        for height in tqdm(self.h[::step]):
            U, V, _, z = stationary_internal_flow_field(
                xs=xs, h_0=height, angle=angle, production=q
            )
            self.Us.append(U)
            self.Vs.append(V)
            self.zs.append(z)


class FiniteVolumeSolver(Solver):
    # A very naive CFL condition, not analytically found at all
    CFL: float = 0.1

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


class UpwindSolver(Solver):
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
