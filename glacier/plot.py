from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

from glacier.solvers import Solver


def animate_glacier(
    solver: Solver,
    interval: float = 100,
    plot_interval: int = 1,
    show: bool = True,
    save_to: Optional[str] = None,
    flow_field: bool = False,
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

    if flow_field:
        solver.calculate_flow_fields(step=plot_interval)
        assert solver.flow_field_step == plot_interval

    def init():
        return

    def update(step, *fargs):
        frame = hs[step]
        for artist in ax.get_children()[0 : 2 if flow_field else 1]:
            artist.remove()
        plt.fill_between(xs, frame, label='current height', color='#0074D9')

        if flow_field:
            # keep = lambda x: not isinstance(x, mpl.patches.FancyArrowPatch)
            ax.patches = []
            U = solver.Us[step]
            V = solver.Vs[step]
            z = solver.zs[step]
            speed = np.sqrt(U * U + V * V)
            ax.streamplot(
                xs,
                z,
                U,
                V,
                linewidth=2 * speed / speed.max() + 1,
                color='white',
                density=1,
            )

    animation = FuncAnimation(
        fig=fig,
        func=update,
        frames=len(hs),
        init_func=init,
        blit=False,
        interval=interval,
        repeat=True,
    )

    if show:
        plt.show()

    if save_to:
        animation.save(save_to + '.gif', dpi=80, writer='imagemagick')

    return fig
