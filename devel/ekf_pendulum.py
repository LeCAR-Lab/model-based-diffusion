import dynamax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jaxtyping import Float, Array
from typing import Callable, NamedTuple

from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, UKFHyperParams
from dynamax.nonlinear_gaussian_ssm import extended_kalman_smoother, unscented_kalman_smoother

# For pretty print of ndarrays
jnp.set_printoptions(formatter={"float_kind": "{:.2f}".format})

# Some parameters
dt = 0.0125
g = 9.8
q_c = 100.0
r = 1.0
num_steps = 400

reward_function = lambda x, u: (- 1.0 - jnp.cos(x[0]))/2.0

# Lightweight container for pendulum parameters
class PendulumParams(NamedTuple):
    initial_state: Float[Array, "state_dim"] = jnp.array([jnp.pi / 2, 0])
    initial_inputs: Float[Array, "num_timesteps input_dim"] = jnp.zeros((num_steps, 1))
    dynamics_function: Callable = lambda x, u: jnp.array([x[0] + x[1] * dt, x[1] + (-g * jnp.sin(x[0])+u[0]) * dt])
    reward_function: Callable = reward_function
    dynamics_covariance: Float[Array, "state_dim state_dim"] = jnp.array([[q_c * dt**3/3, q_c * dt**2/2], [q_c * dt**2/2, q_c * dt]])
    emission_function: Callable = lambda x, u: jnp.append(x, jnp.exp(reward_function(x, u)))
    emission_covariance: Float[Array, "emission_dim"] = jnp.diag(jnp.array([r**2, r**2, 0.001]))

# Pendulum simulation (Särkkä Example 3.7)
def simulate_pendulum(params=PendulumParams(), key=0, num_steps=400):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    # Unpack parameters
    M, N = params.initial_state.shape[0], params.emission_covariance.shape[0]
    f, h = params.dynamics_function, params.emission_function
    Q, R = params.dynamics_covariance, params.emission_covariance

    def _step(carry, x):
        state = carry
        rng, action = x
        rng1, rng2 = jr.split(rng, 2)

        next_state = f(state, action)
        obs = h(next_state, action) + jr.multivariate_normal(rng2, jnp.zeros(N), R)
        return next_state, (next_state, obs)

    rngs = jr.split(key, num_steps)
    _, (states, observations) = lax.scan(_step, params.initial_state, (rngs, params.initial_inputs))

    return states, observations


states, obs = simulate_pendulum(num_steps=num_steps)

def plot_pendulum(time_grid, x_tr, x_obs, x_est=None, est_type=""):
    plt.figure()
    plt.plot(time_grid, x_tr, color="darkgray", linewidth=4, label="True Angle")
    for i in range(x_obs.shape[1]):
        plt.plot(time_grid, x_obs[:, i], "o", fillstyle="none", ms=1.5, label=f"Measurements {i}")
    if x_est is not None:
        for i in range(x_est.shape[1]):
            plt.plot(time_grid, x_est[:, i], linewidth=1.5, label=f"{est_type} Estimate {i}")
    plt.xlabel("Time $t$")
    plt.ylabel("Pendulum angle $x_{1,k}$")
    plt.xlim(0, 5)
    plt.ylim(-6, 6)
    plt.gca().set_aspect(0.3)
    plt.legend(loc=1, borderpad=0.5, handlelength=4, fancybox=False, edgecolor="k")
    plt.savefig(f"../figure/pendulum_{est_type.lower()}.jpg", bbox_inches="tight")

# Create time grid for plotting
time_grid = jnp.arange(0.0, 5.0, step=dt)

# Plot the generated data
plot_pendulum(time_grid, states, obs)

# Compute RMSE
def compute_rmse(y, y_est):
    return jnp.sqrt(jnp.sum((y - y_est) ** 2) / len(y))

# Compute RMSE of estimate and print comparison with
# standard deviation of measurement noise
def compute_and_print_rmse_comparison(y, y_est, R, est_type=""):
    rmse_est = compute_rmse(y, y_est)
    print(f'{f"The RMSE of the {est_type} estimate is":<40}: {rmse_est:.2f}')
    print(f'{"The std of measurement noise is":<40}: {jnp.sqrt(R):.2f}')

pendulum_params = PendulumParams()

# Define parameters for EKF
ekf_params = ParamsNLGSSM(
    initial_mean=pendulum_params.initial_state,
    initial_covariance=jnp.eye(states.shape[-1]) * 0.1,
    dynamics_function=pendulum_params.dynamics_function,
    dynamics_covariance=pendulum_params.dynamics_covariance,
    emission_function=pendulum_params.emission_function,
    emission_covariance=pendulum_params.emission_covariance,
)

obs = obs.at[:, 2].set(1)
ekf_posterior = extended_kalman_smoother(ekf_params, obs, inputs=pendulum_params.initial_inputs)

m_ekf = ekf_posterior.filtered_means
plot_pendulum(time_grid, states, obs, x_est=m_ekf, est_type="EKF")
compute_and_print_rmse_comparison(states[:, 0], m_ekf[:, 0], r, "EKF")

"""
m_ekf = ekf_posterior.smoothed_means[:, 0]
plot_pendulum(time_grid, states[:, 0], obs, x_est=m_ekf, est_type="EKS")
compute_and_print_rmse_comparison(states[:, 0], m_ekf, r, "EKS")
pendulum_params = PendulumParams()

ukf_params = ParamsNLGSSM(
    initial_mean=pendulum_params.initial_state,
    initial_covariance=jnp.eye(states.shape[-1]) * 0.1,
    dynamics_function=pendulum_params.dynamics_function,
    dynamics_covariance=pendulum_params.dynamics_covariance,
    emission_function=pendulum_params.emission_function,
    emission_covariance=pendulum_params.emission_covariance,
)

ukf_hyperparams = UKFHyperParams() # default gives same results as EKF


ukf_posterior = unscented_kalman_smoother(ukf_params, obs, ukf_hyperparams)

m_ukf = ukf_posterior.filtered_means[:, 0]
plot_pendulum(time_grid, states[:, 0], obs, x_est=m_ukf, est_type="UKF")
compute_and_print_rmse_comparison(states[:, 0], m_ukf, r, "UKF")

m_uks = ukf_posterior.smoothed_means[:, 0]
plot_pendulum(time_grid, states[:, 0], obs, x_est=m_uks, est_type="UKS")
compute_and_print_rmse_comparison(states[:, 0], m_uks, r, "UKS")

# Let's see how sensitive UKF is to hyper-params

ukf_hyperparams = UKFHyperParams(alpha=3, beta=3, kappa=3)

ukf_posterior = unscented_kalman_smoother(ukf_params, obs, ukf_hyperparams)

m_ukf = ukf_posterior.filtered_means[:, 0]
plot_pendulum(time_grid, states[:, 0], obs, x_est=m_ukf, est_type="UKF")
compute_and_print_rmse_comparison(states[:, 0], m_ukf, r, "UKF")
"""