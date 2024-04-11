import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define the optimization problem
opti = ca.Opti()

# Define the time horizon and number of control intervals
T = 3.0  # Time horizon
N = 10  # Number of control intervals
dt = T/N  # Time step

# Define the initial and final positions
x0 = np.array([-1.0, 0])
xf = np.array([1.0, 0])

# Define observation terms

# Define the states and controls
x = opti.variable(N+1, 2)  # Position (x, y)
u = opti.variable(N, 2)  # Velocity (vx, vy)
u_max = 1.0  # Maximum velocity

# Define the objective function
objective = 0
Q = np.diag([0.5, 0.5])  # Weight matrix for the states
R = np.diag([0.1, 0.1])  # Weight matrix for the controls
for k in range(N):
    objective += (ca.mtimes([x[k].T, Q, x[k]]) + ca.mtimes([u[k].T, R, u[k]]))

opti.minimize(objective)

# Define the dynamic constraints
for k in range(N):
    opti.subject_to(x[k+1] == x[k] + dt * u[k])  # Dynamics constraint
    opti.subject_to(u[k] <= u_max)  # Maximum velocity constraint

# Define the initial and final boundary conditions
opti.subject_to(x[0] == x0)
opti.subject_to(x[-1] == xf)

# Define the obstacle avoidance constraint
obs_center = np.array([0, 0])  # Center of the obstacle
obs_radius = 0.5  # Radius of the obstacle
for k in range(N+1):
    opti.subject_to(ca.dot(x[k] - obs_center, x[k] - obs_center) >= obs_radius**2)

# Set initial guess
opti.set_initial(x, np.linspace(x0, xf, N+1).T)
opti.set_initial(u, np.zeros((2, N)))

# Set the solver options
p_opts = {"expand": True}
s_opts = {"max_iter": 1000, "tol": 1e-4}
opti.solver("ipopt", p_opts, s_opts)

# Solve the optimization problem
sol = opti.solve()

# Retrieve the optimized states and controls
x_opt = sol.value(x)
u_opt = sol.value(u)


plt.figure(figsize=(8, 6))
plt.plot(x_opt[0, :], x_opt[1, :], '-o')
# plot the obstacle as a circle
circle = plt.Circle(obs_center, obs_radius, color='r', fill=False)
plt.gca().add_artist(circle)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimized Trajectory')
plt.grid(True)
plt.show()