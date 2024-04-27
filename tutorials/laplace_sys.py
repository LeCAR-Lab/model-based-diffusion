import sympy as sp

# Define the variable and the function
x, s = sp.symbols('x s')
f = sp.exp(-x**2) * sp.exp((x-1)**2)

# Calculate the Laplace transform
L_f = sp.laplace_transform(f, x, s, noconds=True)

# Output the result
print(L_f)