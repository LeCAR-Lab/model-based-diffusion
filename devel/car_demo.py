import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define the time horizon and number of control intervals
T = 5.0  # Time horizon
dt = 0.1  # Time step
N = int(T // dt)  # Number of control intervals
Ntraj = 1  # Number of trajectories to diffuse
Ndiff = 30
r_obs = 0.3
obs_center = np.array([
    [-r_obs * 3, r_obs * 2], [-r_obs * 2, r_obs * 2], [-r_obs * 1, r_obs * 2], [0.0, r_obs * 2],
    [0.0, r_obs * 1], [0.0, 0.0], [0.0, -r_obs * 1],
    [-r_obs * 3, -r_obs * 2], [-r_obs * 2, -r_obs * 2], [-r_obs * 1, -r_obs * 2], [0.0, -r_obs * 2],
])
# obs_center = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])  # Center of the obstacle
obs_radius = 0.3  # Radius of the obstacle


## 1st order car dynamics
task = "car"
Q = np.diag([1.0, 0.1, 0.1, 0.1])
R = np.diag([0.1, 0.1])
x0 = np.array([-0.5, 0.0, 0.0, 0.0])
xf = np.array([0.5, 0.0, 0.0, 0.0])
u_max = np.array([1.0, 1.0])
u_min = np.array([-1.0, -1.0])
nx = 4
nu = 2


def dynamics(x, u):
    return ca.vertcat(
        x[3] * ca.sin(x[2])*1.0,  # x_dot
        x[3] * ca.cos(x[2])*1.0,  # y_dot
        x[3] * u[0] * np.pi / 3 * 2.0,  # theta_dot
        u[1] * 6.0,  # v_dot
    )


def stage_cost(x, u):
    return ca.mtimes([(x - xf).T, Q, (x - xf)]) + ca.mtimes([u.T, R, u])


terminal_cost = lambda x: stage_cost(x, ca.DM([0.0, 0.0])) * 10.0


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
    ktm1_xt = (1 - atm1_bar) * np.sqrt(at) / (1 - at_bar)
    ktm1_x0 = (1 - at) * np.sqrt(atm1_bar) / (1 - at_bar)
    var_t = 1 - at_bar
    var_tm1 = (1 - at) * (1 - np.sqrt(atm1_bar)) / (1 - at_bar)

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
        err_yx = yxs[:, k] - x[:, k] * kt
        err_yu = yus[:, k] - u[:, k] * kt
        objective += ca.mtimes([err_yx.T, np.eye(nx), err_yx]) / var_t
        objective += ca.mtimes([err_yu.T, np.eye(nu), err_yu]) / var_t
    objective += terminal_cost(x[:, -1])
    opti.minimize(objective)

    # Define the dynamic constraints
    for k in range(N):
        opti.subject_to(x[:, k + 1] == rk4(dynamics, x[:, k], u[:, k]))
        opti.subject_to(u[:, k] <= u_max)  # Maximum velocity constraint
        opti.subject_to(u[:, k] >= u_min)  # Minimum velocity constraint

    # Define the initial and final boundary conditions
    opti.subject_to(x[:, 0] == x0)
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
    yxs_new = (
        ktm1_x0 * x_opt
        + ktm1_xt * yxs
        + np.random.normal(0, np.sqrt(var_tm1), (nx, N + 1))
    )
    yus_new = (
        ktm1_x0 * u_opt + ktm1_xt * yus + np.random.normal(0, np.sqrt(var_tm1), (nu, N))
    )

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
        xs = xss[j]
        us = uss[j]
        yxs = yxss[j]
        yus = yuss[j]

        # car plot
        ax.plot(xs[0, :], xs[1, :], "b-o", label="Optimized Trajectory", alpha=0.1)
        ax.quiver(
            xs[0, 1:],
            xs[1, 1:],
            np.sin(xs[2, 1:]),
            np.cos(xs[2, 1:]),
            range(N),
            cmap="Blues",
        )
        ax.plot(yxs[0, :], yxs[1, :], "r-", label="Observations", alpha=0.1)
        ax.quiver(
            yxs[0, 1:],
            xs[1, 1:],
            np.sin(yxs[2, 1:]),
            np.cos(yxs[2, 1:]),
            range(N),
            cmap="Reds",
        )
        # ax.quiver(x[0, 1:], x[1, 1:], u[0, :], u[1, :], color='b')
        # ax.quiver(yx[0, :-1], yx[1, :-1], yu[0, :], yu[1, :], range(N), cmap="Reds")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Optimized Trajectory")


# Initialize the trajectory
def generate_xs():
    # xmid = np.array([0.0, (np.random.uniform(1.0, 1.5))*np.sign(np.random.uniform(-1.0, 1.0)), 0.0, 0.0])
    # xmid = np.zeros(nx)
    # xs = np.concatenate([
    #     np.linspace(x0, xmid, N//2+1).T,
    #     np.linspace(xmid, xf, N-N//2).T
    # ], axis=1)
    x_rrt = np.load("../mbd/assets/rrt_path.npy")
    xs = np.zeros((nx, N + 1))
    xs[:2, :N+1] = x_rrt.T
    xs[:2, -1] = x_rrt[-1]
    # xs = xs + np.random.normal(0, 0.1, xs.shape)
    return xs

# Generate the initial trajectory
xss = np.array([generate_xs() for _ in range(Ntraj)])
us = np.zeros((nu, N))
uss = np.tile(us, (Ntraj, 1, 1))
noise_init = 1.0
noise_end = 1e-3
noise_vars = np.exp(np.linspace(np.log(noise_init), np.log(noise_end), Ndiff))

# schedule alpha_bar
alpha_bars = 1 / (1 + np.exp(np.linspace(2.0, -5.0, Ndiff + 1)))
alphas = alpha_bars[1:] / alpha_bars[:-1]

# generate the initial observation
yxss = np.random.normal(0, np.sqrt(1.0 - alpha_bars[0]), (Ntraj, nx, N + 1))
yxss[:, :, 0] = x0
yxss[:, :, -1] = xf
yuss = np.random.normal(0, np.sqrt(1.0 - alpha_bars[0]), (Ntraj, nu, N))

# Optimize the trajectory
fig, ax = plt.subplots()
for i in range(Ndiff):
    atm1_bar = alpha_bars[i + 1]
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
    # plt.savefig(f"../figure/t_{i}.png")
    plt.pause(0.01)
plt.show()
