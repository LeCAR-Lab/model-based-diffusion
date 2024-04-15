import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define the time horizon and number of control intervals
T = 4.0  # Time horizon
dt = 0.1  # Time step
N = int(T // dt)  # Number of control intervals
Ntraj = 1  # Number of trajectories to diffuse
Ndiff = 1
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
# nx = 2
# nu = 2
# def dynamics(x, u):
#     return u
# def stage_cost(x, u):
#     return ca.mtimes([(x - xf).T, Q, x - xf]) + ca.mtimes([u.T, R, u])
# def terminal_cost(x):
#     return stage_cost(x, ca.DM([0.0]))*10.0

## 1st order car dynamics
# task = 'car'
# Q = np.diag([1.0, 0.1, 0.1, 0.1])
# R = np.diag([0.1, 0.1])
# x0 = np.array([-1.0, 0.0, 0.0, 0.0])
# xf = np.array([1.0, 0.0, 0.0, 0.0])
# u_max = np.array([1.0, 1.0])
# u_min = np.array([-1.0, -1.0])
# nx = 4
# nu = 2
# def dynamics(x, u):
#     return ca.vertcat(
#         x[3] * ca.sin(x[2]), # x_dot
#         x[3] * ca.cos(x[2]), # y_dot
#         x[3] * u[0] * np.pi/3, # theta_dot
#         u[1]*6.0 # v_dot
#     )
# def stage_cost(x, u):
#     return ca.mtimes([(x - xf).T, Q, (x - xf)]) + ca.mtimes([u.T, R, u])
# terminal_cost = lambda x: stage_cost(x, ca.DM([0.0, 0.0]))*10.0

## push bar task
task = 'push_bar'
abar = 0.2
nx = 4
nu = 2
T = 5.0  # Time horizon
dt = 0.1  # Time step
N = int(T // dt)  # Number of control intervals
R = np.diag([0.1, 0.3])
x0 = np.array([-0.6, 0.0, np.pi/2, 0.0])  # x, y, theta, p
xf = np.array([0.6, 0.0, 0.0, 0.0])
u_max = np.array([1.0, 1.0])
u_min = np.array([-1.0, -1.0])

def dynamics(x, u):
    theta = x[2]
    p = x[3]
    p_dot = u[0] * 0.3
    k = p / abar
    rbar_dot_norm = u[1] * 1.0 / (1+ca.fabs(k))
    rbar_dot = rbar_dot_norm * ca.vertcat(-ca.sin(theta), ca.cos(theta))
    theta_dot = k * rbar_dot_norm / abar
    return ca.vertcat(rbar_dot, theta_dot, p_dot)

def stage_cost(x, u):
    theta_err = x[2] - xf[2]
    sin_theta_err = ca.sin(theta_err)
    rbar_err = x[0:2] - xf[0:2]
    return ca.mtimes([sin_theta_err.T, sin_theta_err]) + ca.mtimes([rbar_err.T, rbar_err]) + ca.mtimes([u.T, R, u])

def terminal_cost(x):
    return stage_cost(x, ca.DM([0.0, 0.0]))*30.0

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


def optimize_trajectory(xs, us, yxs, yus, at_bar, atm1_bar):
    # Noise related
    at = at_bar / atm1_bar
    kt = np.sqrt(at_bar)
    ktm1_xt = (1-atm1_bar) * np.sqrt(at) / (1-at_bar)
    ktm1_x0 = (1-at)*np.sqrt(atm1_bar) / (1-at_bar)
    var_t = 1 - at_bar
    var_tm1 = (1-at) * (1-np.sqrt(atm1_bar)) / (1-at_bar)

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
        err_yx = yxs[:, k] - x[:, k]*kt
        err_yu = yus[:, k] - u[:, k]*kt
        objective += (
            ca.mtimes([err_yx.T, np.eye(nx), err_yx])
            / var_t
        )
        objective += (
            ca.mtimes([err_yu.T, np.eye(nu), err_yu])
            / var_t
        )
    objective += terminal_cost(x[:, -1])
    opti.minimize(objective)

    # Define the dynamic constraints
    for k in range(N):
        opti.subject_to(x[:, k + 1] == rk4(dynamics, x[:, k], u[:, k]))
        opti.subject_to(u[:, k] <= u_max)  # Maximum velocity constraint
        opti.subject_to(u[:, k] >= u_min)  # Minimum velocity constraint
    if task == "push_bar":
        opti.subject_to(x[3, :] >= -abar)
        opti.subject_to(x[3, :] <= abar)

    # Define the initial and final boundary conditions
    opti.subject_to(x[:, 0] == x0)
    if task in ["inverted_pendulum", "acrobot", "push_bar"]:
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
    s_opts = {"max_iter": 1000, "tol": 1e-4, "print_level": 0}
    opti.solver("ipopt", p_opts, s_opts)

    # Solve the optimization problem
    try:
        sol = opti.solve()        
    except RuntimeError as e:
        print("Failed to solve the optimization problem")
        if "Maximum_Iterations_Exceeded" in str(e):
            print("Maximum iterations exceeded")
            sol = opti.debug

    # sol = opti.solve()
    # Retrieve the optimized states and controls
    x_opt = sol.value(x)
    u_opt = sol.value(u)

    # Generate new observations
    yxs_new = ktm1_x0 * x_opt + ktm1_xt * yxs + np.random.normal(0, np.sqrt(var_tm1), (nx, N + 1))
    yus_new = ktm1_x0 * u_opt + ktm1_xt * yus + np.random.normal(0, np.sqrt(var_tm1), (nu, N))

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

        # push bar plot
        for t in range(x.shape[1]):
            x_t = x[:, t]
            point = get_point(x_t)
            bar_end1, bar_end2 = get_bar_ends(x_t)
            alpha = t / x.shape[1] * 0.9 + 0.1
            ax.plot(
                [bar_end1[0], bar_end2[0]],
                [bar_end1[1], bar_end2[1]],
                "b",
                linewidth=2,
                alpha=alpha,
            )
            ax.plot(point[0], point[1], "ro", alpha=alpha)
        ax.plot(x[0, :], x[1, :], "g-", alpha=1.0)

        # car plot
        # ax.plot(x[0, :], x[1, :], 'b-o', label='Optimized Trajectory', alpha=0.1)
        # ax.quiver(x[0, 1:], x[1, 1:], np.sin(x[2, 1:]), np.cos(x[2, 1:]), range(N), cmap="Blues")
        # ax.plot(yx[0, :], yx[1, :], 'r-', label='Observations', alpha=0.1)
        # ax.quiver(yx[0, 1:], x[1, 1:], np.sin(yx[2, 1:]), np.cos(yx[2, 1:]), range(N), cmap="Reds")
        # ax.quiver(x[0, 1:], x[1, 1:], u[0, :], u[1, :], color='b')
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
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Optimized Trajectory")


# Initialize the trajectory
def generate_xs():
    # xmid = np.array([0.0, (np.random.uniform(1.0, 2.0))*np.sign(np.random.uniform(-1.0, 1.0)), 0.0, 0.0])
    # xmid = np.zeros(nx)
    # xs = np.concatenate([
    #     np.linspace(x0, xmid, N//2+1).T,
    #     np.linspace(xmid, xf, N-N//2).T
    # ], axis=1)
    # xs = np.random.normal(0, 1.0, (nx, N + 1))
    xs = np.linspace(x0, xf, N + 1).T + np.random.normal(0, 1.0, (nx, N + 1))
    if task == "push_bar":
        xs[:2, :] = np.linspace(x0[:2], xf[:2], N + 1).T
        xs[2, :] = 0.0
        xs[3, :] = 0.0
    xs[:, 0] = x0
    xs[:, -1] = xf
    return xs


xss = np.array([generate_xs() for _ in range(Ntraj)])
us = np.zeros((nu, N))
uss = np.tile(us, (Ntraj, 1, 1))
uss = uss + np.clip(np.random.normal(0, 1.0, (Ntraj, nu, N)), -1.0, 1.0)
noise_init = 1.0
noise_end = 1e-3
noise_vars = np.exp(np.linspace(np.log(noise_init), np.log(noise_end), Ndiff))

# schedle alpha_bar
alpha_bars = 1 / (1+np.exp(np.linspace(1.0, -1.0, Ndiff+1)))
alphas = alpha_bars[1:] / alpha_bars[:-1]


yxss = np.random.normal(0, np.sqrt(1.0-alpha_bars[0]), (Ntraj, nx, N + 1))
yxss[:, :, 0] = x0
yxss[:, :, -1] = xf
yuss = np.random.normal(0, np.sqrt(1.0 - alpha_bars[0]), (Ntraj, nu, N))

# Optimize the trajectory
fig, ax = plt.subplots()
for i in range(Ndiff):
    atm1_bar = alpha_bars[i+1]
    at_bar = alpha_bars[i]
    for j in range(Ntraj):
        xs, us, yxs, yus = optimize_trajectory(
            xss[j], uss[j], yxss[j], yuss[j], at_bar, atm1_bar
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
# plt.show()

xs = xss[0]
for i in range(N+1):
    ax.clear()
    # plot bar
    rbar = xs[:2, i]
    theta = xs[2, i]
    p = xs[3, i]
    point = get_point(xs[:, i])
    bar_end1, bar_end2 = get_bar_ends(xs[:, i])
    bar_final_end1, bar_final_end2 = get_bar_ends(xf)
    ax.plot([bar_end1[0], bar_end2[0]], [bar_end1[1], bar_end2[1]], "b", linewidth=5)
    ax.plot(point[0], point[1], "ro")
    ax.plot([bar_final_end1[0], bar_final_end2[0]], [bar_final_end1[1], bar_final_end2[1]], "r--", linewidth=5)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True)
    ax.set_aspect("equal")
    plt.savefig(f"../figure/obs_{i}.png")