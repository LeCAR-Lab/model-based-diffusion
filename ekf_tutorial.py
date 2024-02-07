import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import control as ct

import control.optimal as opt
import control.flatsys as fs

from pvtol import pvtol, pvtol_noisy, plot_results
import pvtol as pvt

"""
set up system
"""

# Find the equiblirum point corresponding to the origin
xe, ue = ct.find_eqpt(
    pvtol,
    np.zeros(pvtol.nstates),
    np.zeros(pvtol.ninputs),
    [0, 0, 0, 0, 0, 0],
    iu=range(2, pvtol.ninputs),
    iy=[0, 1],
)

# Initial condition = 2 meters right, 1 meter up
x0, u0 = ct.find_eqpt(
    pvtol,
    np.zeros(pvtol.nstates),
    np.zeros(pvtol.ninputs),
    np.array([2, 1, 0, 0, 0, 0]),
    iu=range(2, pvtol.ninputs),
    iy=[0, 1],
)

# Extract the linearization for use in LQR design
pvtol_lin = pvtol.linearize(xe, ue)
A, B = pvtol_lin.A, pvtol_lin.B

print("=== Linearized system ===")
print(pvtol, "\n")
print(pvtol_noisy)

"""
test with LQR
"""

# Shoot for 10 cm error in x, 10 cm error in y.  Try to keep the angle
# less than 5 degrees in making the adjustments.  Penalize side forces
# due to loss in efficiency.

Qx = np.diag([100, 10, (180 / np.pi) / 5, 0, 0, 0])
Qu = np.diag([10, 1])
K, _, _ = ct.lqr(A, B, Qx, Qu)

# Compute the full state feedback solution
lqr_ctrl, _ = ct.create_statefbk_iosystem(pvtol, K)  # linear feedback

# Define the closed loop system that will be used to generate trajectories
# input: x_d, u_d, noise
# output:
lqr_clsys = ct.interconnect(
    [pvtol_noisy, lqr_ctrl],
    inplist=lqr_ctrl.input_labels[0 : pvtol.ninputs + pvtol.nstates]
    + pvtol_noisy.input_labels[pvtol.ninputs :],
    inputs=lqr_ctrl.input_labels[0 : pvtol.ninputs + pvtol.nstates]
    + pvtol_noisy.input_labels[pvtol.ninputs :],
    outlist=pvtol.output_labels + lqr_ctrl.output_labels,
    outputs=pvtol.output_labels + lqr_ctrl.output_labels,
)
print("=== Closed loop system ===")
print(lqr_clsys)

"""
set up system for estimation
"""

# Disturbance and noise intensities
Qv = np.diag([1e-2, 1e-2])*1000.0  # control noise
Qw = np.array(
    [[1e-4, 0, 1e-5], [0, 1e-4, 1e-5], [1e-5, 1e-5, 1e-4]]
)*1000.0  # measurement noise
# Initial state covariance
P0 = np.eye(pvtol.nstates)
# Create the time vector for the simulation
Tf = 6
timepts = np.linspace(0, Tf, 20)  # simulate for 6 seconds with 20 steps
# Create representative process disturbance and sensor noise vectors
V = ct.white_noise(timepts, Qv)  # sample with covariance Qv
# V = np.clip(V0, -0.1, 0.1)    # Hold for later
W = ct.white_noise(timepts, Qw)
# plt.plot(timepts, V0[0], 'b--', label="V[0]")
# plt.plot(timepts, V[0], label="V[0]")
# plt.plot(timepts, W[0], label="W[0]")
# plt.legend()
# plt.savefig("noise_VW.png")
# Desired trajectory
xd, ud = xe, ue
# xd = np.vstack([
#     np.sin(2 * np.pi * timepts / timepts[-1]),
#     np.zeros((5, timepts.size))])
# ud = np.outer(ue, np.ones_like(timepts))

# Run a simulation with full state feedback (no noise) to generate a trajectory
uvec = [xd, ud, V, W * 0]  # only control noise
lqr_resp = ct.input_output_response(lqr_clsys, timepts, uvec, x0)
U = lqr_resp.outputs[6:8]  # controller input signals
Y = lqr_resp.outputs[0:3] + W  # noisy output signals (noise in pvtol_noisy)

# Compare to the no noise case
uvec = [xd, ud, V * 0, W * 0]  # no noise
lqr0_resp = ct.input_output_response(lqr_clsys, timepts, uvec, x0)
lqr0_fine = ct.input_output_response(
    lqr_clsys, timepts, uvec, x0, t_eval=np.linspace(timepts[0], timepts[-1], 100)
)
U0 = lqr0_resp.outputs[6:8]
Y0 = lqr0_resp.outputs[0:3]

# Compare the results
plt.figure()
plt.plot(Y0[0], Y0[1], "k--", linewidth=2, label="No disturbances")
plt.plot(lqr0_fine.states[0], lqr0_fine.states[1], "r-", label="Actual")
plt.plot(Y[0], Y[1], "b-", label="Noisy")

plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")
plt.axis("equal")
plt.legend(frameon=False)
plt.savefig("lqr_traj_compare.png")

# plt.figure()
# plot_results(timepts, lqr_resp.states, lqr_resp.outputs[6:8])
# plt.legend(frameon=False)
# plt.savefig("lqr.png")


# Utility functions for making plots
def plot_state_comparison(
    timepts,
    est_states,
    act_states=None,
    obs_states=None,
    estimated_label="$\\hat x_{i}$",
    actual_label="$x_{i}$",
    start=0,
):
    for i in range(sys.nstates):
        plt.subplot(2, 3, i + 1)
        if act_states is not None:
            plt.plot(
                timepts[start:],
                act_states[i, start:],
                "r--",
                label=actual_label.format(i=i),
            )
        if (obs_states is not None) and (i < 3):
            plt.plot(
                timepts[start:],
                obs_states[i, start:],
                "g",
                alpha=0.5,
                label=f'$y_{i}$',
            )
        plt.plot(
            timepts[start:],
            est_states[i, start:],
            "b",
            label=estimated_label.format(i=i),
        )
        plt.legend()
    plt.tight_layout()


# Define a function to plot out all of the relevant signals
def plot_estimator_response(timepts, estimated, U, V, Y, W, start=0):
    # Plot the input signal and disturbance
    for i in [0, 1]:
        # Input signal (the same across all)
        plt.subplot(4, 3, i + 1)
        plt.plot(timepts[start:], U[i, start:], "k")
        plt.ylabel(f"U[{i}]")

        # Plot the estimated disturbance signal
        plt.subplot(4, 3, 4 + i)
        plt.plot(timepts[start:], estimated.inputs[i, start:], "b-", label="est")
        plt.plot(timepts[start:], V[i, start:], "k", label="actual")
        plt.ylabel(f"V[{i}]")

    plt.subplot(4, 3, 6)
    plt.plot(0, 0, "b", label="estimated")
    plt.plot(0, 0, "k", label="actual")
    plt.plot(0, 0, "r", label="measured")
    plt.legend(frameon=False)
    plt.grid(False)
    plt.axis("off")

    # Plot the output (measured and estimated)
    for i in [0, 1, 2]:
        plt.subplot(4, 3, 7 + i)
        plt.plot(timepts[start:], Y[i, start:], "r", label="measured")
        plt.plot(timepts[start:], estimated.states[i, start:], "b", label="measured")
        plt.plot(timepts[start:], Y[i, start:] - W[i, start:], "k", label="actual")
        plt.ylabel(f"Y[{i}]")

    for i in [0, 1, 2]:
        plt.subplot(4, 3, 10 + i)
        plt.plot(timepts[start:], estimated.outputs[i, start:], "b", label="estimated")
        plt.plot(timepts[start:], W[i, start:], "k", label="actual")
        plt.ylabel(f"W[{i}]")
        plt.xlabel("Time [s]")

    plt.tight_layout()


# Create a new system with only x, y, theta as outputs
sys = ct.NonlinearIOSystem(
    updfcn=pvt._noisy_update,
    outfcn=lambda t, x, u, params: x[0:3],
    name="pvtol_noisy",
    states=[f"x{i}" for i in range(6)],
    inputs=["F1", "F2"] + ["Dx", "Dy"],
    outputs=["x", "y", "theta"],
)

"""
Standard Kalman filter
"""

'''
# Standard Kalman filter
# create linear system at the equilibrium point
linsys = sys.linearize(xe, [ue, V[:, 0] * 0])
# print(linsys)
B = linsys.B[:, 0:2]  # only control inputs, no noise inputs
G = linsys.B[:, 2:4]  # only noise inputs, no control inputs
linsys = ct.ss(
    linsys.A,
    B,
    linsys.C,
    0,
    states=sys.state_labels,
    inputs=sys.input_labels[0:2],
    outputs=sys.output_labels,
)
# estimator
# x_hat_k+1 = A x_hat_k + B u_k - L (C x_hat_k - y_k)
# P_k+1 = A P_k A^T - A P_k C^T (C P_k C^T + Q)^-1 C P_k A^T + F Qv F^T
# L = P_k C^T (C P_k C^T + Q)^-1
# where Q is the measurement noise covariance
# and Qv is the process noise covariance
# and F is the disturbance input matrix
# for the following function,
# Qv is the process noise covariance
# and Qw is the measurement noise covariance
# and G is the disturbance input matrix
# and P0 is the initial state covariance
estim = ct.create_estimator_iosystem(linsys, Qv, Qw, G=G, P0=P0)

kf_resp = ct.input_output_response(estim, timepts, [Y, U], X0=[xe, P0.reshape(-1)])
plt.figure()
plot_state_comparison(timepts, kf_resp.outputs, lqr_resp.states)
plt.savefig("kf_linear.png")
'''

"""
Extended Kalman filter
"""

'''
# Define the disturbance input and measured output matrices
F = np.array(
    [
        [0, 0],
        [0, 0],
        [0, 0],
        [1 / pvtol.params["m"], 0],
        [0, 1 / pvtol.params["m"]],
        [0, 0],
    ]
)
C = np.eye(3, 6)
Qwinv = np.linalg.inv(Qw)


# Estimator update law
def estimator_update(t, x, u, params):
    # Extract the states of the estimator
    xhat = x[0 : pvtol.nstates]  # state estimate
    P = x[pvtol.nstates :].reshape(pvtol.nstates, pvtol.nstates)  # prior of P

    # Extract the inputs to the estimator
    y = u[0:3]  # just grab the first three outputs
    u = u[6:8]  # get the inputs that were applied as well

    # Compute the linearization at the current state
    A = pvtol.A(xhat, u)  # A matrix depends on current state
    # A = pvtol.A(xe, ue)       # Fixed A matrix (for testing/comparison)

    # Compute the optimal gain
    L = P @ C.T @ Qwinv

    # Update the state estimate = f(x_hat) - L (z_hat - y)
    xhatdot = pvtol.updfcn(t, xhat, u, params) - L @ (C @ xhat - y)

    # Update the covariance
    Pdot = A @ P + P @ A.T - P @ C.T @ Qwinv @ C @ P + F @ Qv @ F.T

    # Return the derivative
    return np.hstack([xhatdot, Pdot.reshape(-1)])


def estimator_output(t, x, u, params):
    # Return the estimator states
    return x[0 : pvtol.nstates]


ekf = ct.NonlinearIOSystem(
    estimator_update,
    estimator_output,
    states=pvtol.nstates + pvtol.nstates**2,
    inputs=pvtol_noisy.output_labels
    + pvtol_noisy.input_labels[0 : pvtol.ninputs],  # observation y + action u
    outputs=[f"xh{i}" for i in range(pvtol.nstates)],  # state estimate
)

ekf_resp = ct.input_output_response(
    # ekf, timepts, [lqr_resp.states, lqr_resp.outputs[6:8]],
    ekf,
    timepts,
    [Y, np.zeros_like(Y), U],
    X0=[xe, P0.reshape(-1)],
)
plt.figure()
plot_state_comparison(timepts, ekf_resp.outputs, lqr_resp.states)
plt.savefig("ekf.png")
'''

"""
optimization-based estimation: moving horizon estimation
"""

# Define the optimal estimation problem
traj_cost = opt.gaussian_likelihood_cost(sys, Qv, Qw)
init_cost = lambda xhat, x: (xhat - x) @ P0 @ (xhat - x)
oep = opt.OptimalEstimationProblem(sys, timepts, traj_cost, terminal_cost=init_cost)

# Compute the estimate from the noisy signals
est = oep.compute_estimate(Y, U, X0=lqr_resp.states[:, 0])
plt.figure()
plot_state_comparison(timepts, est.states, lqr_resp.states, Y)
plt.savefig("opt.png")

# Plot the response of the estimator
# plt.figure()
# plot_estimator_response(timepts, est, U, V, Y, W)
# plt.savefig("est.png")

'''
Bounded disturbance estimation
'''

'''
V_clipped = np.clip(V, -0.05, 0.05) # clip the control noise
plt.figure()
plt.plot(timepts, V[0], label="V[0]")
plt.plot(timepts, V_clipped[0], label="V[0] clipped")
plt.plot(timepts, W[0], label="W[0]")
plt.legend()
plt.savefig("VW_clipped.png")

uvec = [xe, ue, V_clipped, W]
clipped_resp = ct.input_output_response(lqr_clsys, timepts, uvec, x0)
U_clipped = clipped_resp.outputs[6:8]  # controller input signals
Y_clipped = clipped_resp.outputs[0:3] + W  # noisy output signals

traj_constraint = opt.disturbance_range_constraint(sys, [-0.05, -0.05], [0.05, 0.05])
oep_clipped = opt.OptimalEstimationProblem(
    sys,
    timepts,
    traj_cost,
    terminal_cost=init_cost,
    trajectory_constraints=traj_constraint,
)

est_clipped = oep_clipped.compute_estimate(
    Y_clipped, U_clipped, X0=lqr0_resp.states[:, 0]
)
plt.figure()
plot_state_comparison(timepts, est_clipped.states, lqr_resp.states)
plt.suptitle("MHE with constraints")
plt.tight_layout()
plt.savefig("mhe_clipped.png")

plt.figure()
ekf_unclipped = ct.input_output_response(
    ekf,
    timepts,
    [clipped_resp.states, clipped_resp.outputs[6:8]],
    X0=[xe, P0.reshape(-1)],
)

plot_state_comparison(timepts, ekf_unclipped.outputs, lqr_resp.states)
plt.suptitle("EKF w/out constraints")
plt.tight_layout()
plt.savefig("mhe_unclipped.png")
'''
