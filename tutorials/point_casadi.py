import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define the time horizon and number of control intervals
T = 4.0  # Time horizon
dt = 0.1  # Time step
N = int(T//dt)  # Number of control intervals
Ntraj = 10 # Number of trajectories to diffuse
# obs_center = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, -0.5], [-0.5, -0.5], [-0.5, 0.5]])  # Center of the obstacle
obs_center = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])  # Center of the obstacle
obs_radius = 0.3  # Radius of the obstacle

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


Q = np.diag([1.0, 0.1, 0.1, 0.1])
R = np.diag([0.1, 0.1])
x0 = np.array([-1.0, 0.0, 0.0, 0.0])
xf = np.array([1.0, 0.0, 0.0, 0.0])
u_max = np.array([np.pi/3.0, 6.0])
u_min = np.array([-np.pi/3.0, -6.0])
nx = 4
nu = 2

def dynamics(x, u):
    return ca.vertcat(
        x[3] * ca.sin(x[2]), # x_dot
        x[3] * ca.cos(x[2]), # y_dot
        x[3] * u[0], # theta_dot
        u[1] # v_dot
    )

def rk4(dynamics, x, u):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt/2 * k1, u)
    k3 = dynamics(x + dt/2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def optimize_trajectory(xs, us, yxs, yus, noise_var):
    # Define the optimization problem
    opti = ca.Opti()
    
    # Define the states and controls
    x = opti.variable(nx, N+1)  # Position (x, y)
    u = opti.variable(nu, N)  # Velocity (vx, vy)
    
    # Define the objective function
    objective = 0
    for k in range(N):
        # cost function
        objective += ca.mtimes([(x[:, k] - xf).T, Q, x[:, k] - xf]) + ca.mtimes([u[:, k].T, R, u[:, k]])
        # observation terms
        objective += ca.mtimes([(yxs[:, k] - x[:, k]).T, np.eye(nx), yxs[:, k] - x[:, k]]) / noise_var
        objective += ca.mtimes([(yus[:, k] - u[:, k]).T, np.eye(nu), yus[:, k] - u[:, k]]) / noise_var
    opti.minimize(objective)
    
    # Define the dynamic constraints
    for k in range(N):
        opti.subject_to(x[:, k+1] == rk4(dynamics, x[:, k], u[:, k]))
        opti.subject_to(u[:, k] <= u_max)  # Maximum velocity constraint
        opti.subject_to(u[:, k] >= u_min)  # Minimum velocity constraint
    
    # Define the initial and final boundary conditions
    opti.subject_to(x[:, 0] == x0)
    opti.subject_to(x[:, -1] == xf)
    
    # Define the obstacle avoidance constraint

    for k in range(N+1):
        for i in range(obs_center.shape[0]):
            opti.subject_to(ca.dot(x[:2, k] - obs_center[i], x[:2, k] - obs_center[i]) >= obs_radius**2)
    
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
    yxs_new = x_opt + np.random.normal(0, np.sqrt(noise_var), (nx, N+1))
    yus_new = u_opt + np.random.normal(0, np.sqrt(noise_var), (nu, N))
    
    return x_opt, u_opt, yxs_new, yus_new

def plot_traj(ax, xss, uss, yxss, yuss):
    ax.clear()
    # obstacles
    for i in range(obs_center.shape[0]):
        circle = plt.Circle(obs_center[i, :], obs_radius, color='k', fill=True, alpha=0.5)
        ax.add_artist(circle)
    for j in range(Ntraj):
        x = xss[j]
        u = uss[j]
        yx = yxss[j]
        yu = yuss[j]
        # ax.plot(x[0, :], x[1, :], 'b-o', label='Optimized Trajectory', alpha=0.1)
        # ax.quiver(x[0, 1:], x[1, 1:], u[0, :], u[1, :], color='b')
        # ax.quiver(x[0, 1:], x[1, 1:], np.sin(x[2, 1:]), np.cos(x[2, 1:]), range(N), cmap="Blues")
        ax.plot(yx[0, :], yx[1, :], 'r-', label='Observations', alpha=0.1)
        ax.quiver(yx[0, 1:], x[1, 1:], np.sin(yx[2, 1:]), np.cos(yx[2, 1:]), range(N), cmap="Reds")
        # ax.quiver(yx[0, :-1], yx[1, :-1], yu[0, :], yu[1, :], range(N), cmap="Reds")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    # ax.legend()
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Optimized Trajectory')

# Initialize the trajectory
def generate_xs():
    # xmid = np.array([-1.0, (np.random.uniform(1.5, 2.5))*np.sign(np.random.uniform(-1.0, 1.0)), 0.0, 0.0])
    xmid = np.array([0.0, 0.0, 0.0, 0.0])
    xs = np.concatenate([
        np.linspace(x0, xmid, N//2+1).T,
        np.linspace(xmid, xf, N-N//2).T
    ], axis=1)
    return xs
xss = np.array([generate_xs() for _ in range(Ntraj)])
us = np.zeros((nu, N))
uss = np.tile(us, (Ntraj, 1, 1))
noise_init = 1.0
noise_end = 1e-3
noise_vars = np.exp(np.linspace(np.log(noise_init), np.log(noise_end), 30))
yxss = np.random.normal(0, np.sqrt(noise_vars[0]), (Ntraj, nx, N+1))
yxss[:, :, 0] = x0
yxss[:, :, -1] = xf
yuss = np.random.normal(0, np.sqrt(noise_vars[0]), (Ntraj, nu, N))

# Optimize the trajectory
fig, ax = plt.subplots()
for (i, noise_var) in enumerate(noise_vars):
    for j in range(Ntraj):
        xs, us, yxs, yus = optimize_trajectory(xss[j], uss[j], yxss[j], yuss[j], noise_var)
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