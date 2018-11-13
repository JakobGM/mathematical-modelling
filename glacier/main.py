import numpy as np

from glacier.physics import GlacierParameters
from glacier.solvers import Solver

L = 500
n_x = L + 1
xs = np.linspace(0, L, n_x)
H = 25
alpha = np.radians(3)
t_end = 10
h_0 = 50

upwind_scheme = False
steady_state = False
plot_initial = False

if steady_state:
    q_0 = 1
    glacier = GlacierParameters(
        xs=xs, alpha=alpha, q_0=0, x_s=xs[-1] * 0.3, x_f=xs[-1] * 0.6, h_0=h_0
    )
else:
    glacier = GlacierParameters(
        xs=xs, alpha=alpha, q_0=1e0, x_s=xs[-1] * 0, x_f=xs[-1] * 0.9, h_0=h_0
    )
    q = glacier.q.unscaled * 3600 * 24 * 365
    left_height = h_0
    h_0 = np.zeros(len(xs))
    h_0[0] = left_height
    glacier = GlacierParameters(xs=xs, q=q, h_0=h_0, alpha=alpha)

if plot_initial:
    glacier.plot()

if upwind_scheme:
    solver = Solver(glacier=glacier, name='upwind')
    solver.solve(t_end, method="upwind")
else:
    solver = Solver(glacier=glacier, name='finite_volume')
    solver.solve(t_end, method="finite volume", save_every=100)

solver.calculate_flow_fields(save_every=20)
solver.animate(plot_step=10, show=True)
# animate_glacier(solver, interval=1, plot_interval=10, flow_field=False)
