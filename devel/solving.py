import numpy as np
from scipy.optimize import brentq

# Constants
A = 0.1  # example value for A
C = 1.0  # example value for C

# Define the function
def equation(x):
    term1 = 1 / np.sqrt(x)
    term2 = 0.5 * (1 - x) / np.sqrt(1 - C * x)
    return term1 - term2 - A

# Find a root within a specified interval
# Note: Choose the interval such that it is within the valid range of the function
lower_bound = 0.8
upper_bound = 0.9
print("Lower bound:", equation(lower_bound))
print("Upper bound:", equation(upper_bound))

# Solve the equation
root = brentq(equation, lower_bound, upper_bound)

print("Root:", root)

# for i in range(10):
#     A = 