from typing import Optional

from glacier.solvers import Solver

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots()
    ax.set_title('Glacier Animation')
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(hs.min(), hs.max() * 2)

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

    # Create line segment updated in each frame
    line, = plt.plot(xs, hs[0], label='current height')

    ax.legend()

    def init():
        line.set_ydata([np.nan] * len(xs))
        return (line,)

    def update(frame, *fargs):
        # ln.set_data(xs, frame)
        line.set_ydata(frame)
        return (line,)

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
        animation.save(save_to, dpi=80, writer='imagemagick')

    if show:
        plt.show()

    return fig
