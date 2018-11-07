from plot import animate_glacier
from solvers import FiniteVolumeSolver, GlacierParameters

import numpy as np


def test_animate_glacier():
    xs = np.arange(start=0, stop=100).astype(float)
    glacier = GlacierParameters(
        xs=xs, alpha=np.radians(3), h_0=50, q_0=0.0001, x_s=50, x_f=70
    )
    solver = FiniteVolumeSolver(glacier=glacier)
    solver.solve(t_end=1, delta_t=0.01)
    animate_glacier(solver=solver, show=False)
