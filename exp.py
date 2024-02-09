import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import control as ct

import control.optimal as opt
import control.flatsys as fs

from pvtol import pvtol, pvtol_noisy, plot_results
import pvtol as pvt
# import ekf_tutorial as ekf

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
Qv = np.diag([1e-2, 1e-2])*10.0  # control noise
Qw = np.array(
    [[1e-4, 0, 1e-5], [0, 1e-4, 1e-5], [1e-5, 1e-5, 1e-4]]
)*10000.0  # measurement noise
# Initial state covariance
P0 = np.eye(pvtol.nstates)*1.0
Tf = 6  # simulate for 6 seconds
timepts = np.linspace(0, Tf, 20)  # simulate for 6 seconds with 20 steps
x0 = np.array([2.0, 1.0, 0, 0, 0, 0])  # initial state
C = np.eye(3, 6) # output matrix
Q_all = np.array([100, 10, (180 / np.pi) / 5, 0, 0, 0, 10, 1])
Q_extended = np.stack([Q_all] * 20).T

# Define the optimal estimation problem
traj_cost = opt.gaussian_likelihood_cost(sys, Qv, Qw)
init_cost = lambda xhat, x: (xhat - x) @ P0 @ (xhat - x)
oep = opt.OptimalEstimationProblem(sys, timepts, traj_cost, terminal_cost=init_cost)

def plot_traj(traj: np.ndarray):
    x = traj[0]
    y = traj[1]
    theta = traj[2]
    # plt.figure()
    # plot arrows based on (x, y) with angle theta
    plt.plot(x, y)
    plt.quiver(x, y, np.sin(theta), np.cos(theta))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.xlim([-3, 3])
    plt.ylim([-2, 2])
    # plt.savefig('traj.png')
    print('Trajectory plotted')

def refine_traj(traj: np.ndarray): 
    # Generate the input signal
    Y = traj[:3]
    U = traj[-2:]
    # Compute the estimate from the noisy signals
    # ekf_resp = ct.input_output_response(
    #     ekf.ekf,
    #     timepts,
    #     traj, 
    #     # [Y, np.zeros_like(Y), U],
    #     X0=[ekf.xe, P0.reshape(-1)],
    # )
    est = oep.compute_estimate(Y, U, X0=x0)
    # refine the trajectory
    # traj_hat =ekf_resp.outputs
    # print(est.inputs.shape)
    U_hat = U + est.inputs
    traj_hat = np.concatenate([est.states, U_hat], axis=0)
    return traj_hat

def main():
    # ground truth trajectory from LQR
    traj_true = np.load('lqr_resp.npy')

    # generate random trajectory
    # generate a line from (2,1) to (0,0)
    # x = np.linspace(2, 0, 20)
    # y = np.linspace(1, 0, 20)
    # theta = np.zeros_like(x)
    # # add noise to the trajectory
    # x += np.random.normal(0, 0.5, 20)
    # y += np.random.normal(0, 0.5, 20)
    # theta += np.random.normal(0, np.pi, 20) % (2*np.pi)
    # Y = np.vstack([x, y, theta])
    # traj = np.concatenate([Y, np.zeros_like(Y), np.zeros((2, 20))], axis=0)
    traj = traj_true.copy()
    traj[:3] += np.random.multivariate_normal([0, 0, 0], Qw, 20).T
    traj[-2:] += np.random.multivariate_normal([0, 0], Qv, 20).T

    plt.figure()
    plot_traj(traj)
    plt.savefig(f'traj_0.png')

    for itr in range(10):
        # calculate new trajectory esitimation
        traj_hat = refine_traj(traj)
        # traj_hat[:, 0] = x0
        # traj_hat[:, -1] = np.zeros_like(x0)
        traj_delta_dyn = traj_hat - traj
        traj_delta_cost = - 2 * Q_extended * traj

        # print(np.sqrt((traj_delta_dyn**2).mean()))
        # print(np.sqrt((traj_delta_cost**2).mean()))
        # print(np.sqrt((traj**2).mean()))

        traj = traj + (traj_delta_cost*0.01 + traj_delta_dyn*0.1)
        # traj[:6, 0] = x0
        plt.figure()
        plot_traj(traj)
        plt.savefig(f'traj_{itr+1}.png')

        # calculate new cost function guidance 

        # calculate trajectory

if __name__ == '__main__':
    main()