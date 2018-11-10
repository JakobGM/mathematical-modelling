from typing import Optional

from glacier.solvers import Solver

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np


def animate_glacier(
    solver: Solver,
    interval: float = 100,
    plot_interval: int = 1,
    show: bool = True,
    save_to: Optional[str] = None,
):
    glacier = solver.glacier
    xs = glacier.xs.unscaled
    hs = solver.h[::plot_interval]

    # Create figure used for animation
    fig, ax = plt.subplots(subplot_kw={'autoscale_on': False})
    ax.set_title('Glacier Animation')
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(hs.min(), 1.4 * hs.max())

    # Plot initial height
    ax.plot(
        xs,
        hs[0],
        alpha=0.5,
        color='red',
        linestyle='--',
        label='initial height',
    )

    # Plot steady state height
    steady_height = solver.glacier.generate_steady_state_height()
    ax.plot(
        xs,
        steady_height,
        alpha=0.5,
        color='green',
        linestyle='--',
        label='steady state height',
    )

    # Plot accumulation rate
    q = glacier.q.unscaled.copy()

    # Scale hot and cold areas seperately to [-1, 1]
    q[q < 0] /= -q.min()
    q[q > 0] /= q.max()
    assert q.max() == 1
    assert q.min() == -1

    # Negate q in order to get proper coolwarm colorscheme
    background = [-q]
    ax.imshow(
        background,
        aspect='auto',
        vmin=-1,
        vmax=1,
        extent=[xs.min(), xs.max(), hs.min(), 2 * hs.max()],
        alpha=0.4,
        cmap='coolwarm',
    )
    ax.set_aspect(10)

    # Create line segment updated in each frame
    filled_glacier = plt.fill_between(
        xs, hs[0], label='current height', color='#0074D9'
    )

    ax.legend()

    def init():
        return

    def update(frame, *fargs):
        latest_child = ax.get_children()[0]
        latest_child.remove()
        plt.fill_between(xs, frame, label='current height', color='#0074D9')

    animation = FuncAnimation(
        fig=fig,
        func=update,
        frames=hs,
        init_func=init,
        blit=False,
        interval=interval,
        repeat=True,
    )

    if save_to:
        animation.save(save_to + '.gif', dpi=80, writer='imagemagick')

    if show:
        plt.show()

    return fig
