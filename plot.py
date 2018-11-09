from typing import Any

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

import numpy as np


def animate_glacier(solver: Any, interval: float = 100, plot_interval=1, show: bool = True):
    glacier = solver.glacier
    xs = glacier.xs.unscaled
    hs = solver.h

    fig, ax = plt.subplots()
    ax.set_title('Glacier Animation')
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(hs.min(), hs.max()*2)
    line, = plt.plot(xs, hs[0])

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
        frames=hs[::plot_interval, :],
        init_func=init,
        blit=False,
        interval=interval,
        repeat=True,
    )

    if show:
        plt.show()

    return fig
