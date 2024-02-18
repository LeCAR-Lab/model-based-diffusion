# Generate data for control problem.
import numpy as np

# Form and solve control problem.
import cvxpy as cp

# Plot results.
import matplotlib.pyplot as plt

np.random.seed(1)
n = 4
m = 2
T = 50
dt = 0.1
# contionuse A
Ac = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])
Bc = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0]
])
Zc = np.block([
    [Ac, Bc],
    [np.zeros((m, n)), np.zeros((m, m))]
])
Z = np.eye(n + m) + Zc * dt
A = Z[:n, :n]
B = Z[:n, n:]
Q = np.eye(n)
R = np.eye(m) * 0.1

x = cp.Variable((n, T + 1))
u = cp.Variable((m, T))

x_err = x[:, :-1] - np.array([1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
u_err = u[:, :]
cost = 0
constr = []
for t in range(T):
    # cost
    cost += cp.quad_form(x_err[:, t], Q) + cp.quad_form(u_err[:, t], R)
    # dynamics
    constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]
    # constrains stay away from circle at (0,0) with radius 0.3
    constr += [(x[0, t] ** 2 + x[1, t] ** 2) >= 0.3 ** 2]
    # control constraints
    constr += [u[:, t] <= np.array([1.0, 1.0])]
    constr += [u[:, t] >= -np.array([1.0, 1.0])]
# initial condition
x_0 = np.array([-1.0, 0.0, 0.0, 0.0])
constr += [x[:, 0] == x_0]

# sums problem objectives and concatenates constraints.
problem = cp.Problem(cp.Minimize(cost), constr)
result = problem.solve(verbose=False)

f = plt.figure()
# plot circle at (0,0) with radius 0.3
circle = plt.Circle((0, 0), 0.3, color='r', fill=False)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().add_artist(circle)
plt.plot(x.value[0, :].T, x.value[1, :].T, 'o')
plt.plot(0, 0, 'ro')
plt.legend()
plt.savefig("../figure/cvx.png")
# close figure
plt.close(f)
# plot x - t
f = plt.figure()
plt.plot(x.value.T)
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.savefig("../figure/cvx_x.png")
plt.close(f)
# plot u - t
f = plt.figure()
plt.plot(u.value.T)
plt.xlabel('t')
plt.ylabel('u')
plt.legend()
plt.savefig("../figure/cvx_u.png")
plt.close(f)