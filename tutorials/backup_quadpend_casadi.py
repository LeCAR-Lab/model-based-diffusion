import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define the time horizon and number of control intervals
T = 4.0  # Time horizon
dt = 0.1  # Time step
N = int(T // dt)  # Number of control intervals
Ntraj = 8  # Number of trajectories to diffuse
# obs_center = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, -0.5], [-0.5, -0.5], [-0.5, 0.5]])  # Center of the obstacle
# obs_center = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])  # Center of the obstacle
obs_center = np.array([])  # Center of the obstacle
obs_radius = 0.3  # Radius of the obstacle

## 1st order point mload dynamics
# Q = np.diag([0.5, 0.5])  # Weight matrix for the states
# R = np.diag([0.1, 0.1])  # Weight matrix for the controls
# x0 = np.array([-0.5, 0])  # Initial position
# xf = np.array([0.5, 0])  # Final position
# u_max = 1.0  # Maximum velocity
# u_min = -1.0
# nx = 4
# nu = 2
# def dynamics(x, u):
#     return u

## 1st order car dynamics
# Q = np.diag([1.0, 0.1, 0.1, 0.1])
# R = np.diag([0.1, 0.1])
# x0 = np.array([-1.0, 0.0, 0.0, 0.0])
# xf = np.array([1.0, 0.0, 0.0, 0.0])
# u_max = np.array([np.pi/3.0, 6.0])
# u_min = np.array([-np.pi/3.0, -6.0])
# nx = 4
# nu = 2
# def dynamics(x, u):
#     return ca.vertcat(
#         x[3] * ca.sin(x[2]), # x_dot
#         x[3] * ca.cos(x[2]), # y_dot
#         x[3] * u[0], # theta_dot
#         u[1] # v_dot
#     )
# def stage_cost(x, u):
#     return ca.mtimes([(x - xf).T, Q, x - xf]) + ca.mtimes([u.T, R, u])
# terminal_cost = lambda x: stage_cost(x, ca.DM([0.0]))

## quad pendulum
task = "quad_pendulum"
g = 9.81
nx = 8
nu = 2
mlift = 0.5
mload = 0.1
rlift = 0.25
lrope = 0.5
Jlift = 0.004
fric = 0.01
x0 = np.zeros(nx)
# x0[0] = -3.0
xf = np.zeros(nx)
# xf[0] = 3.0
u_max = 3.0 * mlift * g * np.ones(nu)
u_min = np.zeros(nu)
u_hover = (mlift + mload) * g * np.ones(nu) * 0.5
weights = (0.01, 0.05, 5.)
Q_T = (10., 10., 1., 1., 1., 1., 1., 1.)
def get_mass_matrix(q):
    phi = q[-1]

    M_q = ca.vertcat(
        ca.horzcat(mlift + mload, 0.0, 0.0, mload * lrope * ca.cos(phi)),
        ca.horzcat(0.0, mlift + mload, 0.0, mload * lrope * ca.sin(phi)),
        ca.horzcat(0.0, 0.0, Jlift, 0.0),
        ca.horzcat(
            mload * lrope * ca.cos(phi),
            mload * lrope * ca.sin(phi),
            0.0,
            mload * lrope * lrope,
        ),
    )

    return M_q


def get_mass_inv(q):
    phi = q[-1]

    a = mlift + mload
    b = mload * lrope * ca.cos(phi)
    c = mload * lrope * ca.sin(phi)
    d = mload * lrope * lrope
    den = (mload * lrope) ** 2.0 - a * d
    M_inv = ca.vertcat(
        ca.horzcat((c * c - a * d) / (a * den), -(b * c) / (a * den), 0.0, b / den),
        ca.horzcat(-(b * c) / (a * den), (b * b - a * d) / (a * den), 0.0, c / den),
        ca.horzcat(0.0, 0.0, 1.0 / Jlift, 0.0),
        ca.horzcat(b / den, c / den, 0.0, -a / den),
    )
    return M_inv


def kinetic(q, q_dot, get_mass_matrix):
    return 0.5 * ca.dot(q_dot, get_mass_matrix(q) @ q_dot)


def potential(q, mlift, g, mload, lrope):
    return mlift * g * q[1] + mload * g * (q[1] - lrope * ca.cos(q[-1]))


def lag(q, q_dot, get_mass_matrix, mlift, g, mload, lrope):
    return kinetic(q, q_dot, get_mass_matrix) - potential(q, mlift, g, mload, lrope)


def dL_dq(q, q_dot, get_mass_matrix, mlift, g, mload, lrope):
    return ca.jacobian(lag(q, q_dot, get_mass_matrix, mlift, g, mload, lrope), q)


def dynamics(x, u):
    q = x[:4]
    q_dot = x[4:]
    M_dot = ca.jacobian(get_mass_matrix(q), q) @ q_dot
    M_inv = get_mass_inv(q)

    torque_fric_pole = -fric * (q_dot[-1] - q_dot[-2])
    F_q = ca.vertcat(
        -ca.sum1(u) * ca.sin(q[2]),
        ca.sum1(u) * ca.cos(q[2]),
        (u[0] - u[1]) * rlift - torque_fric_pole,
        torque_fric_pole,
    )

    q_ddot = M_inv @ (
        F_q + dL_dq(q, q_dot, get_mass_matrix, mlift, g, mload, lrope) - (M_dot @ q_dot)
    )

    return ca.vertcat(q_dot, q_ddot)

def stage_cost(x, u):
    delta = x - xf
    pos_cost = ca.dot(delta[:3], delta[:3]) + (1. + ca.cos(x[3]))
    ctrl_cost = ca.dot(u - u_hover, u - u_hover)
    stage_cost = weights[0] * pos_cost + weights[1] * ctrl_cost
    return stage_cost

def terminal_cost(x):
    delta = x - xf
    term_cost = weights[2] * ca.dot(delta, ca.vertcat(*[Q_T[i] * delta[i] for i in range(nx)]))
    return term_cost


## inverted pendulum dynamics
# task = 'inverted_pendulum'
# Q = np.diag([1.0, 0.1])
# R = np.diag([0.1])
# x0 = np.array([0.0, 0.0])
# xf = np.array([np.pi, 0.0])
# u_max = np.array([1.0])
# u_min = np.array([-1.0])
# nx = 2
# nu = 1
# g = 1.5
# def dynamics(x, u):
#     return ca.vertcat(
#         x[1], # theta_dot
#         u[0]-g*ca.sin(x[0]) # v_dot
#     )*3.0
# def stage_cost(x, u):
#     x_new = ca.vertcat(ca.cos(x[0]), x[1])
#     xf_new = ca.vertcat(ca.cos(xf[0]), xf[1])
#     return ca.mtimes([(x_new - xf_new).T, Q, x_new - xf_new]) + ca.mtimes([u.T, R, u])
# def terminal_cost(x):
#     return stage_cost(x, ca.DM([0.0]))*30.0

## acrobot dynamics
# task = 'acrobot'
# Q = np.diag([1.0, 1.0, 0.1, 0.1])
# R = np.diag([0.1])
# x0 = np.array([0.0, 0.0, 0.0, 0.0])
# xf = np.array([np.pi, 0.0, 0.0, 0.0])
# u_max = np.array([1.0])
# u_min = np.array([-1.0])
# nx = 4
# nu = 1
# g = 0.5
# def dynamics(x, u):
#     theta2_dot = (u[0] - ca.sin(x[1])*g*ca.cos(x[0]) - ca.sin(x[1])*ca.cos(x[0])*ca.sin(x[1]))/(1.0 + ca.sin(x[1])**2)
#     theta1_dot = ca.cos(x[1])*theta2_dot - ca.cos(x[0])*ca.sin(x[1])*g
#     return ca.vertcat(
#         x[1], # theta1_dot
#         x[2], # theta2_dot
#         theta1_dot, # theta1_dot_dot
#         theta2_dot # theta2_dot_dot
#     )*3.0
# def stage_cost(x, u):
#     ry0 = 0.5*ca.cos(x[0])
#     ry1 = 0.5*ca.cos(x[0]) + 0.5*ca.cos(x[0]+x[1])
#     x_new = ca.vertcat(ry0, ry1, x[2], x[3])
#     ryf0 = 0.5*ca.cos(xf[0])
#     ryf1 = 0.5*ca.cos(xf[0]) + 0.5*ca.cos(xf[0]+xf[1])
#     xf_new = ca.vertcat(ryf0, ryf1, xf[2], xf[3])
#     return ca.mtimes([(x_new - xf_new).T, Q, x_new - xf_new]) + ca.mtimes([u.T, R, u])
# def terminal_cost(x):
#     return stage_cost(x, ca.DM([0.0]))*100.0


def rk4(dynamics, x, u):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def optimize_trajectory(xs, us, yxs, yus, noise_var):
    # Define the optimization problem
    opti = ca.Opti()

    # Define the states and controls
    x = opti.variable(nx, N + 1)  # Position (x, y)
    u = opti.variable(nu, N)  # Velocity (vx, vy)

    # Define the objective function
    objective = 0
    for k in range(N):
        # cost function
        objective += stage_cost(x[:, k], u[:, k])
        # observation terms
        objective += (
            ca.mtimes([(yxs[:, k] - x[:, k]).T, np.eye(nx), yxs[:, k] - x[:, k]])
            / noise_var
        )
        objective += (
            ca.mtimes([(yus[:, k] - u[:, k]).T, np.eye(nu), yus[:, k] - u[:, k]])
            / noise_var
        )
    objective += terminal_cost(x[:, -1])
    opti.minimize(objective)

    # Define the dynamic constraints
    for k in range(N):
        opti.subject_to(x[:, k + 1] == rk4(dynamics, x[:, k], u[:, k]))
        opti.subject_to(u[:, k] <= u_max)  # Maximum velocity constraint
        opti.subject_to(u[:, k] >= u_min)  # Minimum velocity constraint

    # Define the initial and final boundary conditions
    opti.subject_to(x[:, 0] == x0)
    if task in ["inverted_pendulum", "acrobot"]:
        pass
    else:
        opti.subject_to(x[:, -1] == xf)

    # Define the obstacle avoidance constraint
    for i in range(obs_center.shape[0]):
        for k in range(N + 1):
            opti.subject_to(
                ca.dot(x[:2, k] - obs_center[i], x[:2, k] - obs_center[i])
                >= obs_radius**2
            )

    # Set initial guess
    opti.set_initial(x, xs)
    opti.set_initial(u, us)

    # Set the solver options
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "tol": 1e-4}
    opti.solver("ipopt", p_opts, s_opts)

    # Solve the optimization problem
    sol = opti.solve()

    # Retrieve the optimized states and controls
    x_opt = sol.value(x)
    u_opt = sol.value(u)

    # Generate new observations
    yxs_new = x_opt + np.random.normal(0, np.sqrt(noise_var), (nx, N + 1))
    yus_new = u_opt + np.random.normal(0, np.sqrt(noise_var), (nu, N))

    return x_opt, u_opt, yxs_new, yus_new


def plot_traj(ax, xss, uss, yxss, yuss):
    ax.clear()
    # obstacles
    for i in range(obs_center.shape[0]):
        circle = plt.Circle(
            obs_center[i, :], obs_radius, color="k", fill=True, alpha=0.5
        )
        ax.add_artist(circle)
    for j in range(Ntraj):
        x = xss[j]
        u = uss[j]
        yx = yxss[j]
        yu = yuss[j]
        # ax.plot(x[0, :], x[1, :], 'b-o', label='Optimized Trajectory', alpha=0.1)
        # ax.quiver(x[0, 1:], x[1, 1:], u[0, :], u[1, :], color='b')
        # ax.quiver(x[0, 1:], x[1, 1:], np.sin(x[2, 1:]), np.cos(x[2, 1:]), range(N), cmap="Blues")
        # ax.plot(yx[0, :], yx[1, :], 'r-', label='Observations', alpha=0.1)
        # ax.quiver(yx[0, 1:], x[1, 1:], np.sin(yx[2, 1:]), np.cos(yx[2, 1:]), range(N), cmap="Reds")
        # ax.quiver(yx[0, :-1], yx[1, :-1], yu[0, :], yu[1, :], range(N), cmap="Reds")

        # pendulum plot
        # ax.scatter(np.arange(N+1), yx[0, :], c=range(N+1), cmap='Reds')
        # ax.plot(yx[0, :], 'r', alpha=0.3)
        # ax.plot(yx[1, :], 'r--', alpha=0.1)

        # acrobot plot
        # ax.scatter(np.arange(N+1), yx[0, :], c=range(N+1), cmap='Reds')
        # ax.scatter(np.arange(N+1), yx[1, :], c=range(N+1), cmap='Blues')
        # ax.plot(yx[0, :], 'r', alpha=0.3)
        # ax.plot(yx[1, :], 'b', alpha=0.3)
        # ry0 = -np.cos(yx[0, :])
        # ry1 =-np.cos(yx[0, :]) - np.cos(yx[0, :]+yx[1, :])
        # rx0 = np.sin(yx[0, :])
        # rx1 = np.sin(yx[0, :]) + np.sin(yx[0, :]+yx[1, :])
        # ax.scatter(np.arange(N+1), rx0, c=range(N+1), cmap='Reds')
        # ax.scatter(np.arange(N+1), rx1, c=range(N+1), cmap='Reds')
        # ax.scatter(np.arange(N+1), ry0, c=range(N+1), cmap='Reds')
        # ax.scatter(np.arange(N+1), ry1, c=range(N+1), cmap='Blues')
        # for k in range(N):
        #     # get color from Reds depending on the value of k
        #     ax.plot([0.0, rx0[k]], [0.0, ry0[k]], 'r', alpha=((k/N)*0.8+0.2))
        #     ax.plot([rx0[k], rx1[k]], [ry0[k], ry1[k]], 'b', alpha=((k/N)*0.8+0.2))
        # ax.plot(x[0, :], 'b', alpha=1.0)
        # ax.plot(x[1, :], 'g', alpha=1)
        # ax.plot(x[2, :], 'r', alpha=1)
        # ax.plot(x[3, :], 'k', alpha=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    # ax.set_ylim(-4, 4)
    # ax.legend()
    # ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Optimized Trajectory")


# Initialize the trajectory
def generate_xs():
    # xmid = np.array([-1.0, (np.random.uniform(1.5, 2.5))*np.sign(np.random.uniform(-1.0, 1.0)), 0.0, 0.0])
    # xmid = np.zeros(nx)
    # xs = np.concatenate([
    #     np.linspace(x0, xmid, N//2+1).T,
    #     np.linspace(xmid, xf, N-N//2).T
    # ], axis=1)
    xs = np.zeros((nx, N + 1))
    return xs


xss = np.array([generate_xs() for _ in range(Ntraj)])
us = np.zeros((nu, N))
uss = np.tile(us, (Ntraj, 1, 1))
noise_init = 1.0
noise_end = 1e-3
noise_vars = np.exp(np.linspace(np.log(noise_init), np.log(noise_end), 30))
yxss = np.random.normal(0, np.sqrt(noise_vars[0]), (Ntraj, nx, N + 1))
yxss[:, :, 0] = x0
yxss[:, :, -1] = xf
yuss = np.random.normal(0, np.sqrt(noise_vars[0]), (Ntraj, nu, N))

# Optimize the trajectory
fig, ax = plt.subplots()
for i, noise_var in enumerate(noise_vars):
    for j in range(Ntraj):
        xs, us, yxs, yus = optimize_trajectory(
            xss[j], uss[j], yxss[j], yuss[j], noise_var
        )
        xss[j] = xs
        uss[j] = us
        yxss[j] = yxs
        yuss[j] = yus
    # xs, us, yxs, yus = optimize_trajectory(xs, us, yxs, yus, noise_var)
    # Plot the optimized trajectory
    plot_traj(ax, xss, uss, yxss, yuss)
    plt.savefig(f"../figure/t_{i}.png")
    plt.pause(0.01)
plt.show()
