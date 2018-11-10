from glacier.plot import animate_glacier
from glacier.solvers import GlacierParameters, UpwindSolver

import numpy as np


def test_animate_glacier():
    xs = np.arange(start=0, stop=1000).astype(float)
    glacier = GlacierParameters(
        xs=xs, alpha=np.radians(3), h_0=40, q_0=70, x_s=400, x_f=700
    )
    solver = UpwindSolver(glacier=glacier)
    solver.solve(t_end=1, delta_t=0.01)
    animate_glacier(solver=solver, show=False)
