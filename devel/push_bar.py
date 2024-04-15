import numpy as np
from matplotlib import pyplot as plt

T = 4.0
dt = 0.1
N = int(T/dt)

abar = 0.2
nx = 4
nu = 2

def dynamics(x, u):
    theta = x[2]
    p = x[3]
    p_dot = u[0] * 0.3
    k = p / abar
    rbar_dot_norm = u[1] / (1+np.abs(k))
    rbar_dot = rbar_dot_norm * np.array([-np.sin(theta), np.cos(theta)])
    theta_dot = k * rbar_dot_norm / abar
    return np.concatenate([rbar_dot, np.array([theta_dot, p_dot])])

def get_bar_ends(x):
    rbar = x[0:2]
    theta = x[2]
    bar_end1 = rbar + np.array([np.cos(theta), np.sin(theta)]) * abar 
    bar_end2 = rbar - np.array([np.cos(theta), np.sin(theta)]) * abar
    return bar_end1, bar_end2

def get_point(x):
    rbar = x[0:2]
    theta = x[2]
    p = x[3]
    return rbar + np.array([np.cos(theta), np.sin(theta)]) * p

def plot_traj(ax, xss, uss):
    ax.clear()
    for j in range(xss.shape[0]):
        xs = xss[j]

        for t in range(xs.shape[0]):
            x = xs[t]
            point = get_point(x)
            bar_end1, bar_end2 = get_bar_ends(x)
            alpha = t/xs.shape[0]*0.5 + 0.5
            ax.plot([bar_end1[0], bar_end2[0]], [bar_end1[1], bar_end2[1]], "b", linewidth=5, alpha=alpha)
            ax.plot(point[0], point[1], "ro", alpha=alpha)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("push bar")


x = np.array([0.0, 0.0, 0.0, 0.0]) # x, y, theta, p
u = np.array([0.0, 1.0])

# simulate the system
xs = np.zeros((N+1, nx))
xs[0] = x
for i in range(1, N+1):
    x = x + dynamics(x, u) * dt
    xs[i] = x
# plot the trajectory
fig, ax = plt.subplots()
plot_traj(ax, xs[None], None)
plt.show()