import numpy as np
import matplotlib.pyplot as plt

Q = np.diag([0.5, 1])

def f(x):
    return 0.5 * np.dot(x - np.array([1, 0]), Q.dot(x - np.array([1, 0])))

def grad_f(x):
    return Q.dot(x - np.array([1, 0]))

def hessian_f(x):
    return Q

A = np.array([1.0, -1.0])
b = -1.0

def c(x):
    return np.dot(A, x) - b

def grad_c(x):
    return A

def plot_landscape():
    Nsamp = 20
    Xsamp, Ysamp = np.meshgrid(np.linspace(-4, 4, Nsamp), np.linspace(-4, 4, Nsamp))
    Zsamp = np.zeros((Nsamp, Nsamp))
    for j in range(Nsamp):
        for k in range(Nsamp):
            Zsamp[j, k] = f(np.array([Xsamp[j, k], Ysamp[j, k]]))
    plt.contour(Xsamp, Ysamp, Zsamp)

    xc = np.linspace(-4, 3, Nsamp)
    plt.plot(xc, xc + 1, "y")

plot_landscape()

def La(x, lmbda, rho):
    p = max(0, c(x))
    return f(x) + lmbda * p + (rho / 2) * (p ** 2)

def newton_solve(x0, lmbda, rho):
    # initial guess
    x = x0
    p = max(0, c(x))
    C = np.zeros(2)
    if c(x) >= 0:
        C = grad_c(x)
    g = grad_f(x) + (lmbda + rho * p) * C.T
    while np.linalg.norm(g) >= 1e-8:
        H = hessian_f(x) + rho * np.dot(C.T, C)
        dx = -np.linalg.solve(H, g)
        
        x += dx
        
        # augmented Lagrangian extra term
        p = max(0, c(x))
        # gradient of the inequality constraint
        C = np.zeros(2)
        if c(x) >= 0:
            C = grad_c(x)
        g = grad_f(x) + (lmbda + rho * p) * C.T

    return x

xguess = np.array([-3.0, 2.0])
lmbda_guess = np.array([0.0])
rho = 1.0

plot_landscape()
plt.plot(xguess[0], xguess[1], "rx")
# save the current figure
plt.savefig("landscape.png")
plt.cla()

xnew = newton_solve(xguess, lmbda_guess[-1], rho)
lmbda_new = max(0, lmbda_guess[-1] + rho * c(xnew))
xguess = np.column_stack([xguess, xnew])
lmbda_guess = np.append(lmbda_guess, lmbda_new)
rho = 10 * rho

plot_landscape()
plt.plot(xguess[0, :], xguess[1, :], "rx")
plt.savefig("landscape2.png")