import jax
import chex
from jax import lax
from jax import numpy as jnp
from flax import struct
import matplotlib.pyplot as plt

import env
import vis

# global static parameters
n_state: int = 4
n_action: int = 2
horizon: int = 100
diffuse_step = 20
diffuse_substeps = 10


@struct.dataclass
class Params:
    # environment parameters
    dt: float = 0.1
    r_obs: float = 0.2
    init_state: jnp.ndarray = jnp.array([-1.0, 0.0, 0.0, 0.0])
    goal_state: jnp.ndarray = jnp.array([1.0, 0.0, 0.0, 0.0])

    # diffuser parameters
    noise_std: float = 0.1
    langevin_eps: float = 0.1


def get_reward(
    x_traj: jnp.ndarray,
    u_traj: jnp.ndarray,
    params: Params,
) -> jnp.ndarray:
    def get_running_cost(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        Q = jnp.diag(jnp.array([1.0, 1.0, 0.1, 0.1]))
        R = jnp.eye(n_action) * 0.1
        x_err = x - jnp.array([1.0, 0.0, 0.0, 0.0])
        return (x_err @ Q @ x_err + u @ R @ u) / 2.0

    running_cost = jax.vmap(get_running_cost)(x_traj, u_traj).sum()
    return -running_cost.sum()


def get_logpd_scan(x_traj, u_traj, params):
    def step(state, input):
        x_hat, cov_hat, logpd = state
        u_prev, x_current = input
        A = env.get_A(x_hat, params)
        B = env.get_B(x_hat, params)
        Q = B @ B.T * params.noise_std**2
        R = jnp.eye(n_state) * params.noise_std**2
        x_pred = A @ x_hat + B @ u_prev
        cov_pred = A @ cov_hat @ A.T + Q
        K = cov_pred @ jnp.linalg.inv(cov_pred + R)
        x_hat = x_pred + K @ (x_current - x_pred)
        cov_hat = (jnp.eye(n_state) - K) @ cov_pred
        y_cov_pred = cov_pred + R
        logpd += -0.5 * (
            jnp.log(2 * jnp.pi) * n_state
            + jnp.linalg.slogdet(y_cov_pred)[1]
            + (x_current - x_pred).T @ jnp.linalg.inv(y_cov_pred) @ (x_current - x_pred)
        )
        return (x_hat, cov_hat, logpd), None

    x_hat = x_traj[0]
    cov_hat = jnp.eye(n_state) * 0.0
    logpd = 0.0
    initial_state = (x_hat, cov_hat, logpd)

    inputs = (u_traj, x_traj)
    (x_hat, cov_hat, logpd), _ = lax.scan(step, initial_state, inputs)
    return logpd  # Return only the logpd component of the final state


def update_traj(
    x_traj: jnp.ndarray,
    u_traj,
    params: Params,
    rng: chex.PRNGKey,
) -> jnp.ndarray:
    reward_grad = jax.grad(get_reward, argnums=[1, 2])
    reward_grad_x, reward_grad_u = reward_grad(x_traj, u_traj, params)

    logpd_grad = jax.grad(get_logpd_scan, argnums=[1, 2])
    logpd_grad_x, logpd_grad_u = logpd_grad(
        x_traj, u_traj, params
    )


    logp_dyn = get_logpd_scan(x_traj, u_traj, params)
    reward_scale_large = 1 / params.noise_std**2
    reward_scale_small = 1.0
    reward_scale = jnp.where(logp_dyn > -100, reward_scale_large, reward_scale_small)
    grad_x = logpd_grad_x + reward_grad_x * reward_scale

    grad_u = logpd_grad_u + reward_grad_u * reward_scale
    eps = params.langevin_eps
    rng, rng_x, rng_u = jax.random.split(rng, 3)
    x_traj_new = (
        x_traj
        + eps * grad_x
        + jnp.sqrt(2 * eps) * jax.random.normal(rng_x, grad_x.shape)
    )
    x_traj_new = x_traj_new.at[0].set(
        params.init_state
    )  # NOTE: do not add noise to the initial state
    x_traj_new = x_traj_new.at[-1].set(
        params.goal_state
    )  # NOTE: do not add noise to the final state
    u_traj_new = (
        u_traj
        + eps * grad_u
        + jnp.sqrt(2 * eps) * jax.random.normal(rng_u, grad_u.shape)
    )

    return x_traj_new, u_traj_new


def main():
    # check NaN with jax
    # jax.config.update("jax_debug_nans", True)
    # init params
    params = Params()
    rng = jax.random.PRNGKey(1)

    # schedule langevin episilon
    langevin_eps_schedule = jnp.linspace(1.0, 0.5, diffuse_substeps) * 1e-5  # 1e-5
    # schedule global noise (perturbation noise)
    noise_std_init = 2e-1
    noise_std_final = 1e-2
    # plan in exponential space
    scale = 1.0
    noise_std_schedule = (
        jnp.exp(
            jnp.linspace(
                jnp.log(noise_std_init * scale),
                jnp.log(noise_std_final * scale),
                diffuse_step,
            )
        )
        / scale
    )
    # noise_std_schedule = jnp.linspace(noise_std_init, noise_std_final, diffuse_step)

    # init trajectory
    rng, rng_x, rng_u = jax.random.split(rng, 3)
    x_traj = jax.random.normal(rng_x, (horizon, n_state))
    u_traj = jax.random.normal(rng_u, (horizon, n_action))
    x_traj = x_traj.at[0].set(params.init_state)

    # run MBD
    log_info = {
        "d_step": 0,
        "sub_step": 0,
        "total_step": 0,
        "noise_std": jnp.zeros(diffuse_step * diffuse_substeps),
        "langevin_eps": jnp.zeros(diffuse_step * diffuse_substeps),
        "logpd": jnp.zeros(diffuse_step * diffuse_substeps),
        "logp_reward": jnp.zeros(diffuse_step * diffuse_substeps),
    }
    update_traj_jit = jax.jit(update_traj)
    get_logpd_scan_jit = jax.jit(get_logpd_scan)
    for d_step in range(diffuse_step):
        noise_std = noise_std_schedule[d_step]
        params = params.replace(noise_std=noise_std)
        for sub_step in range(diffuse_substeps):
            langevin_eps = (
                langevin_eps_schedule[sub_step] * (params.noise_std / 5e-3) ** 2
            )
            params = params.replace(langevin_eps=langevin_eps)

            rng, rng_traj = jax.random.split(rng)
            x_traj, u_traj = update_traj_jit(x_traj, u_traj, params, params, rng_traj)
            logpd = get_logpd_scan_jit(x_traj, u_traj, params, params)
            logp_reward = get_reward(x_traj, u_traj, params)

            # log info
            log_info["d_step"] = d_step
            log_info["sub_step"] = sub_step
            log_info["total_step"] = d_step * diffuse_substeps + sub_step
            log_info["logpd"] = (
                log_info["logpd"].at[d_step * diffuse_substeps + sub_step].set(logpd)
            )
            log_info["logp_reward"] = (
                log_info["logp_reward"]
                .at[d_step * diffuse_substeps + sub_step]
                .set(logp_reward)
            )
            log_info["noise_std"] = (
                log_info["noise_std"]
                .at[d_step * diffuse_substeps + sub_step]
                .set(noise_std)
            )
            log_info["langevin_eps"] = (
                log_info["langevin_eps"]
                .at[d_step * diffuse_substeps + sub_step]
                .set(langevin_eps)
            )
            jax.debug.print(
                "d_step = {d_step}, substep = {substep}, logp_dynamic = {x:.2f}, logp_reward = {y:.2f}",
                d_step=d_step,
                substep=sub_step,
                x=logpd,
                y=logp_reward,
            )

            # rollout dynamics to get real trajectory
            x_traj_real = jnp.zeros((horizon, n_state))
            x_traj_real = x_traj_real.at[0].set(params.init_state)
            for t in range(1, horizon):
                x_traj_real = x_traj_real.at[t].set(
                    env.get_A(x_traj_real[t - 1], params) @ x_traj_real[t - 1]
                    + env.get_B(x_traj_real[t - 1], params) @ u_traj[t - 1]
                )

            # visualize trajectory
            vis.plot_traj(
                x_traj,
                u_traj,
                x_traj_real,
                log_info,
                f"traj_{d_step*diffuse_substeps+sub_step}",
            )


if __name__ == "__main__":
    main()