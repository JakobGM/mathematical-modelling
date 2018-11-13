from typing import Optional, TYPE_CHECKING


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

if TYPE_CHECKING:
    from glacier.solvers import Solver


def animate_glacier(
    solver: 'Solver',
    interval: float = 100,
    plot_interval: int = 1,
    show: bool = True,
    save_to: Optional[str] = None,
    flow_field: bool = False,
):
    glacier = solver.glacier
    xs = glacier.xs.unscaled
    hs = solver.h

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
    first_glacier = plt.fill_between(
        xs, hs[0], label='current height', color='#0074D9'
    )
    artists = [first_glacier, None]

    ax.legend()

    if flow_field:
        global_max_speed = 0
        for U, V in zip(solver.Us, solver.Vs):
            max_speed = (U ** 2 + V ** 2).flatten().max()
            if max_speed > global_max_speed:
                global_max_speed = max_speed

    def init():
        return

    def update(step, *fargs):
        frame = hs[step]
        glacier_artist = artists[0]
        artists[0] = plt.fill_between(
            xs, frame, label='current height', color='#0074D9'
        )
        glacier_artist.remove()

        if flow_field and step in solver.flow_field_steps:
            # keep = lambda x: not isinstance(x, mpl.patches.FancyArrowPatch)
            ax.patches = []
            flow_step = np.where(solver.flow_field_steps == step)[0][0]
            U = solver.Us[flow_step]
            V = solver.Vs[flow_step]
            z = solver.zs[flow_step]
            speed = np.sqrt(U * U + V * V)
            new_stream_artist = ax.streamplot(
                xs,
                z,
                U,
                V,
                linewidth=(1000 * speed / global_max_speed).clip(
                    min=0.3, max=4
                ),
                color='white',
            )
            stream_artist = artists[1]
            artists[1] = new_stream_artist
            if stream_artist:
                stream_artist.lines.remove()

    animation = FuncAnimation(
        fig=fig,
        func=update,
        frames=np.arange(start=0, stop=len(hs), step=plot_interval),
        init_func=init,
        blit=False,
        interval=interval,
        repeat=True,
    )

    if show:
        plt.show()

    if save_to:
        filename = save_to + '.gif'
        print(f'Saving animation to "{filename}"')
        animation.save(filename, dpi=80, writer='imagemagick')

    return fig
