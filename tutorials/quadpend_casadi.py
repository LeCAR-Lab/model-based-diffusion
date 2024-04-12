import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define the time horizon and number of control intervals
T = 4.0  # Time horizon
dt = 0.1  # Time step
N = int(T // dt)  # Number of control intervals
Ntraj = 1  # Number of trajectories to diffuse
# obs_center = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, -0.5], [-0.5, -0.5], [-0.5, 0.5]])  # Center of the obstacle
obs_center = np.array([[0.0, -0.15-1.0], [0.0, -0.15], [0.0, -0.15+1.0]])  # Center of the obstacle
# obs_center = np.array([[0.0, 0.0]])  # Center of the obstacle
# obs_center = np.array([])  # Center of the obstacle
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

## quadrotor pendulum dynamics
task = "quadpend"
nx = 10
nu = 3
g = 1.0
mload = 0.1
mlift = 0.05
Jlift = 0.1
lrope = 0.3
Q = np.diag([1.0, 1.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.0, 0.1, 0.1])
R = np.diag([0.1, 0.1, 0.0])
x0 = np.array([-1.0, 0.0, 0.0, -1.0, -lrope, 0.0, 0.0, 0.0, 0.0, 0.0])
xf = np.array([1.0, 0.0, 0.0, 1.0, -lrope, 0.0, 0.0, 0.0, 0.0, 0.0])
u_max = np.array([3.0*(mload+mlift)*g, 1.0, 100.0])
u_min = np.array([0.0, -1.0, -100.0])
u_hover = np.array([(mload+mlift)*g, 0.0, mload*g])
def dynamics(x, u):
    rlift = x[:2]
    theta = x[2]
    rload = x[3:5]
    vlift = x[5:7]
    omega = x[7]
    vload = x[8:10]

    load2lift = rload - rlift
    e_load2lift = load2lift / ca.norm_2(load2lift)
    f_rope = e_load2lift * u[2]

    rlift_dot = vlift
    theta_dot = omega
    rload_dot = vload
    vlift_dot = ca.vertcat(0.0, -g) + ca.vertcat(ca.sin(theta) * u[0], ca.cos(theta) * u[0])/mlift + f_rope/mlift
    omega_dot = u[1]/Jlift
    vload_dot = -f_rope/mload + ca.vertcat(0.0, -g)

    return ca.vertcat(rlift_dot, theta_dot, rload_dot, vlift_dot, omega_dot, vload_dot)

def stage_cost(x, u):
    return ca.mtimes([(x - xf).T, Q, (x - xf)]) + ca.mtimes([(u - u_hover).T, R, (u - u_hover)])

def terminal_cost(x):
    return stage_cost(x, ca.DM([0.0]))*100.0

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
        if task == "quadpend":
            rlift = x[:2, k+1]
            rload = x[3:5, k+1]
            opti.subject_to(ca.dot(rload - rlift, rload - rlift) == lrope**2)
        

    # Define the initial and final boundary conditions
    opti.subject_to(x[:, 0] == x0)
    if task in ["inverted_pendulum", "acrobot", "quadpend"]:
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
            if task == "quadpend":
                rload = x[3:5, k]
                opti.subject_to(
                    ca.dot(rload - obs_center[i], rload - obs_center[i])
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

        # quadpend
        rlift = yx[:2, :]
        theta = yx[2, :]
        rload = yx[3:5, :]
        ax.quiver(rlift[0, :], rlift[1, :], np.sin(theta), np.cos(theta), range(N+1), cmap="Reds")
        ax.scatter(rload[0, :], rload[1, :], c=range(N+1), cmap='Blues')


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
    ax.set_aspect('equal')
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
    # xs = np.tile(x0, (N + 1, 1)).T
    xs = np.linspace(x0, xf, N + 1).T
    return xs


xss = np.array([generate_xs() for _ in range(Ntraj)])
# us = np.zeros((nu, N))
us = np.tile(u_hover, (N, 1)).T
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

x = xss[0]
for i in range(N+1):
    ax.clear()
    rlift = x[:2, :]
    theta = x[2, :]
    rload = x[3:5, :]
    for j in range(obs_center.shape[0]):
        circle = plt.Circle(
            obs_center[j, :], obs_radius, color="k", fill=True, alpha=0.5
        )
        ax.add_artist(circle)
    ax.plot(rlift[0, :], rlift[1, :], 'r--', alpha=0.3)
    ax.plot(rload[0, :], rload[1, :], 'b--', alpha=0.3)
    ax.quiver(rlift[0, i], rlift[1, i], np.sin(theta[i]), np.cos(theta[i]), color='r')
    ax.scatter(rload[0, i], rload[1, i], color = 'b')
    ax.plot([rlift[0, i], rload[0, i]], [rlift[1, i], rload[1, i]], 'k')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)
    plt.savefig(f"../figure/obs_{i}.png")
plt.show()