import numpy as np

import matplotlib.pyplot as plt

from solvers import GlacierParameters


def stationary_internal_flow_field(xs, h_0, angle, q_0, x_s, x_f):
    alpha = np.radians(angle)

    glacier = GlacierParameters(xs=xs, x_s=x_s, x_f=x_f, q_0=q_0, h_0=h_0, alpha=alpha)

    xs = glacier.xs.scaled
    hs = glacier.h_0.scaled

    zs = np.linspace(0, np.max(hs), xs.shape[0])
    
    u = lambda i, z: glacier.kappa*(hs[i]**(glacier.m+1) - (hs[i] - z)**(glacier.m + 1))/(glacier.m + 1)
    
    f = lambda i, z: hs[i]**(glacier.m + 1)*z + ((hs[i] - z)**(glacier.m+2) - hs[i]**(glacier.m+2))/(glacier.m+2)

    f_derivative = lambda i, z, dx: (f(i+1, z) - f(i-1, z))/(2*dx)

    v = lambda i, z, dx: -glacier.kappa/(glacier.m + 1)*f_derivative(i, z, dx)

    U = np.zeros((xs.shape[0], zs.shape[0]))
    V = np.zeros((xs.shape[0], zs.shape[0]))

    for i in range(1, xs.shape[0]-1):
        for j in range(1, zs.shape[0] - 1):
            U[i, j] = u(i, zs[j])
            V[i, j] = v(i, zs[j], xs[1] - xs[0])
        U[i, np.greater(zs, np.ones((zs.shape))*hs[i])] = 0
        V[i, np.greater(zs, np.ones((zs.shape))*hs[i])] = 0
    
    
    return U.T, V.T, glacier, zs


def plot_internal_flow_field(glacier, zs, U, V):
    h = glacier.h_0.unscaled
    x = glacier.xs.unscaled
    z = zs*glacier.H

    #Horizontal velocity scaler
    U_scaling = glacier.Q*glacier.L/glacier.H

    #Vertical velocity scaler
    V_scaling = glacier.Q

    inc = 1
    fig = glacier.plot(show=False)
    axes = fig.axes
    axes[0].streamplot(x, z, U*U_scaling, V*V_scaling, color=np.sqrt((np.power(U,2)+np.power(V, 2))), cmap='summer')
    plt.show()

angle = 5
h_0 = 40
xs = np.linspace(0, 500, 1000)
x_s = 100
x_f = 400
q_0 = 0.05

U, V, glacier, zs = stationary_internal_flow_field(xs, h_0, angle, q_0, x_s, x_f)
plot_internal_flow_field(glacier, zs, U, V)