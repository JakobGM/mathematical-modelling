from solvers import UpwindSolver, FiniteVolumeSolver, GlacierParameters
from plot import animate_glacier
import numpy as np

L = 1000
n_x = L + 1
xs = np.linspace(0, L, n_x)
alpha = np.radians(3)
t_end = 40

steady_state = False
upwind_scheme = True
plot_initial = False

if steady_state:
    q_0 = 1
    glacier = GlacierParameters(xs=xs, alpha=alpha, q_0=1e0, x_s=xs[-1]*0.3, x_f=xs[-1]*0.6, h_0=50)
else:
    q = np.zeros(len(xs))
    h_0 = np.zeros(len(xs))
    H = 50

    # h_0[0:int(len(xs)/2)] = H * np.sin(np.linspace(0, np.pi, int(len(xs)/2)))
    log_xs = np.log(xs + 1)
    log_xs = log_xs / max(log_xs) * H
    h_0[0 : int(L / 2) + 1] = log_xs[int(L / 2) : None : -1]

    q[0 : int(L / 10)] = 0.01
    q[int(L / 10) : int(L / 2)] = np.linspace(1, -5, num=(int(L / 2) - int(L / 10)))

    glacier = GlacierParameters(xs=xs, q=q, h_0=h_0, alpha=alpha)

if plot_initial:
    glacier.plot()

if upwind_scheme:
    solver = UpwindSolver(glacier=glacier)
else:
    solver = FiniteVolumeSolver(glacier=glacier)

solver.solve(t_end)
animate_glacier(solver, interval=10)
