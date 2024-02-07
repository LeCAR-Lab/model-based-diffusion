import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import control as ct

import control.optimal as opt
import control.flatsys as fs

from pvtol import pvtol, pvtol_noisy, plot_results
import pvtol as pvt
import ekf_tutorial as ekf

# Create a new system with only x, y, theta as outputs
sys = ct.NonlinearIOSystem(
    updfcn=pvt._noisy_update,
    outfcn=lambda t, x, u, params: x[0:3],
    name="pvtol_noisy",
    states=[f"x{i}" for i in range(6)],
    inputs=["F1", "F2"] + ["Dx", "Dy"],
    outputs=["x", "y", "theta"],
)

# Disturbance and noise intensities
Qv = np.diag([1e-2, 1e-2])*1000.0  # control noise
Qw = np.array(
    [[1e-4, 0, 1e-5], [0, 1e-4, 1e-5], [1e-5, 1e-5, 1e-4]]
)*1000.0  # measurement noise
# Initial state covariance
P0 = np.eye(pvtol.nstates)
Tf = 6  # simulate for 6 seconds
timepts = np.linspace(0, Tf, 20)  # simulate for 6 seconds with 20 steps
x0 = np.array([2.0, 1.0, 0, 0, 0, 0])  # initial state
C = np.eye(3, 6) # output matrix

# Define the optimal estimation problem
traj_cost = opt.gaussian_likelihood_cost(sys, Qv, Qw)
init_cost = lambda xhat, x: (xhat - x) @ P0 @ (xhat - x)
oep = opt.OptimalEstimationProblem(sys, timepts, traj_cost, terminal_cost=init_cost)

def refine_traj(traj: np.ndarray): 
    # Generate the input signal
    Y = traj[:, :3]
    U = traj[:, -2:]
    # Compute the estimate from the noisy signals
    est = oep.compute_estimate(Y, U, X0=x0)
    # refine the trajectory
    traj_hat = est.states
    return traj_hat

def main():
    # ground truth trajectory from LQR
    traj_true = np.load('lqr_resp.npy')

    # generate random trajectory

    for itr in range(10):
        # calculate new trajectory esitimation

        # calculate new cost function guidance 

        # calculate trajectory

if __name__ == '__main__':
    main()