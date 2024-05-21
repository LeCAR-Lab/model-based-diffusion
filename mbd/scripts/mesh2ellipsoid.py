import numpy as np
from scipy.optimize import minimize
from stl import mesh  # You might need to install numpy-stl with `pip3 install numpy-stl`

def ellipsoid_fit(X, *params):
    # This function should define the distance from the ellipsoid to the points
    # An ellipsoid equation in general form is (x/a)^2 + (y/b)^2 + (z/c)^2 = 1,
    # where a, b, and c are the radii along the x, y, and z axes.
    a, b, c = params
    x, y, z = X[:,0], X[:,1], X[:,2]
    return np.mean((x/a)**2 + (y/b)**2 + (z/c)**2 - 1)**2

def main():
    # Load your STL file
    your_mesh = mesh.Mesh.from_file('/home/pcy/Research/code/diffuser-planer/mbd/assets/unitree_g1/assets/head_link.STL')
    points = np.array(your_mesh.points).reshape(-1, 3)

    # Initial guess for parameters a, b, c
    initial_guess = [1, 1, 1]

    # Perform the minimization
    result = minimize(ellipsoid_fit, initial_guess, args=(points,))
    print("Fitted ellipsoid parameters (a, b, c):", result.x)

if __name__ == "__main__":
    main()