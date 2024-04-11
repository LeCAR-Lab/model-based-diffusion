import numpy as np
from pydrake.all import MathematicalProgram, Solve, Variable, CustomConstraint, SnoptSolver

# Define the initial and final points
x0 = np.array([0., 0.])
xf = np.array([1., 1.])

# Define the time horizon and number of time steps
T = 10.0
N = 100

# Create a mathematical program
prog = MathematicalProgram()

# Define the state and control variables
x = prog.NewContinuousVariables(2, N+1, "x")
v = prog.NewContinuousVariables(2, N, "v")

# Set the initial and final state constraints
prog.AddLinearConstraint(x[:, 0] == x0)
prog.AddLinearConstraint(x[:, -1] == xf)

# Define the dynamics constraint
def dynamics_constraint(vars):
    x, v = np.split(vars, 2)
    dt = T / N
    return x[1:] - x[:-1] - dt * v

prog.AddConstraint(CustomConstraint(dynamics_constraint, lb=np.zeros(2*N), ub=np.zeros(2*N)))

# Define the cost function (minimize control effort)
prog.AddQuadraticCost(np.sum(v**2))

# Set the IPOPT solver
solver = SnoptSolver()

# Solve the optimization problem
result = Solve(prog, solver)

# Retrieve the optimized states and controls
x_opt = result.GetSolution(x)
v_opt = result.GetSolution(v)

# Print the optimized trajectory
print("Optimized trajectory:")
for i in range(N+1):
    print(f"Time step {i}: x = {x_opt[:, i]}, v = {v_opt[:, i] if i < N else 'N/A'}")