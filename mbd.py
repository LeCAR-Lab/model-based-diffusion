import jax
import chex
from jax import lax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt

# global static parameters
n_state: int = 6
n_action: int = 2
horizon: int = 50


@struct.dataclass
class MBDParams:
    diffuse_step: int = 10
    noise_std: float = 1.0
    langevin_eps: float = 0.1
    # reward_scale: float = 100.0


@struct.dataclass
class EnvParams:
    dt: float = 0.1
    mass: float = 1.0
    inertia: float = 1.0
    init_state: jnp.ndarray = jnp.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_state: jnp.ndarray = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def get_A(x: jnp.ndarray, env_params: EnvParams) -> jnp.ndarray:
    return jnp.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            # [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ) * env_params.dt + jnp.eye(n_state)


def get_B(x: jnp.ndarray, env_params: EnvParams) -> jnp.ndarray:
    theta = x[2]
    return (
        jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                # [1.0 / env_params.mass * jnp.cos(theta), 0.0],
                # [-1.0 / env_params.mass * jnp.sin(theta), 0.0],
                # [0.0, 1.0 / env_params.inertia],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        * env_params.dt
    )


def get_reward(
    x_traj: jnp.ndarray,
    u_traj: jnp.ndarray,
    mdb_params: MBDParams,
    env_params: EnvParams,
) -> jnp.ndarray:
    dist_rew = (
        1.0
        - jnp.clip(
            ((x_traj[:, :2] - env_params.goal_state[:2]) ** 2).sum(axis=1), 0.0, 5.0
        )
        / 5.0
    )
    dist2center = ((x_traj[:, :2] - jnp.array([0.0, 0.0])) ** 2).sum(axis=1)
    obs_rew = jnp.clip((dist2center - 0.2) / 0.1, -2.0, 1.0)
    final_rew = (
        1.0
        - jnp.clip(((x_traj[-1, :2] - env_params.goal_state[:2]) ** 2).sum(), 0.0, 5.0)
        / 5.0
    )
    u_rew = 1.0 - jnp.clip((u_traj**2).sum(axis=1), 0.0, 1.0)
    # dist2goal_normed = dist2goal / (mdb_params.noise_std**2)
    return (dist_rew.sum() + u_rew.sum() * 0.1 + obs_rew.sum() * 3.0)
# / (
#         mdb_params.noise_std
    # )  # + obs_rew.sum()*10.0


def get_logp_dynamics(
    x_traj: jnp.ndarray,
    u_traj: jnp.ndarray,
    mdb_params: MBDParams,
    env_params: EnvParams,
) -> jnp.ndarray:
    # tau: (n_step, n_dim)
    x_hat_traj = jnp.zeros_like(x_traj)
    x_hat_traj = x_hat_traj.at[0].set(x_traj[0])
    cov_hat = jnp.eye(n_state) * 0.0
    logp_dynamics = 0.0

    for t in range(1, horizon):
        # parse state and action
        u_prev = u_traj[t - 1]
        # get state prediction
        A = get_A(x_hat_traj[t - 1], env_params)  # NOTE: A is discrete time
        B = get_B(x_hat_traj[t - 1], env_params)
        Q = B @ B.T * mdb_params.noise_std**2
        R = jnp.eye(n_state) * mdb_params.noise_std**2
        x_pred = A @ x_hat_traj[t - 1] + B @ u_prev
        # get prediction covariance
        cov_pred = A @ cov_hat @ A.T + Q
        # get optimal prediction feedback matrix
        K = cov_pred @ jnp.linalg.inv(cov_pred + R)
        # update state and covariance
        x_hat = x_pred + K @ (x_traj[t] - x_pred)
        x_hat_traj = x_hat_traj.at[t].set(x_hat)
        cov_hat = (jnp.eye(n_state) - K) @ cov_pred
        # update logp_dynamics
        y_cov_pred = (
            cov_pred + R
        )  # NOTE: y_cov_pred is the covariance of the observation
        logp_dynamics += -0.5 * (
            jnp.log(2 * jnp.pi) * n_state
            + jnp.linalg.slogdet(y_cov_pred)[1]
            + (x_traj[t] - x_pred).T @ jnp.linalg.inv(y_cov_pred) @ (x_traj[t] - x_pred)
        )

    return logp_dynamics  # , x_traj_filtered


def get_logp_dynamics_scan(x_traj, u_traj, mdb_params, env_params):
    def step(state, input):
        x_hat, cov_hat, logp_dynamics = state
        u_prev, x_current = input
        A = get_A(x_hat, env_params)
        B = get_B(x_hat, env_params)
        Q = B @ B.T * mdb_params.noise_std**2
        R = jnp.eye(n_state) * mdb_params.noise_std**2
        x_pred = A @ x_hat + B @ u_prev
        cov_pred = A @ cov_hat @ A.T + Q
        K = cov_pred @ jnp.linalg.inv(cov_pred + R)
        x_hat = x_pred + K @ (x_current - x_pred)
        cov_hat = (jnp.eye(n_state) - K) @ cov_pred
        y_cov_pred = cov_pred + R
        logp_dynamics += -0.5 * (
            jnp.log(2 * jnp.pi) * n_state
            + jnp.linalg.slogdet(y_cov_pred)[1]
            + (x_current - x_pred).T @ jnp.linalg.inv(y_cov_pred) @ (x_current - x_pred)
        )
        return (x_hat, cov_hat, logp_dynamics), None

    x_hat = x_traj[0]
    cov_hat = jnp.eye(n_state) * 0.0
    logp_dynamics = 0.0
    initial_state = (x_hat, cov_hat, logp_dynamics)

    inputs = (u_traj, x_traj)
    (x_hat, cov_hat, logp_dynamics), _ = lax.scan(step, initial_state, inputs)
    return logp_dynamics  # Return only the logp_dynamics component of the final state


def get_next_traj(
    x_traj: jnp.ndarray,
    u_traj,
    mdb_params: MBDParams,
    env_params: EnvParams,
    rng: chex.PRNGKey,
) -> jnp.ndarray:
    def get_reward_wo_x0(x0, x_traj_future, u_traj, mdb_params, env_params):
        x_traj = jnp.concatenate([x0[None], x_traj_future], axis=0)
        return get_reward(x_traj, u_traj, mdb_params, env_params)

    reward_grad = jax.grad(get_reward_wo_x0, argnums=[1, 2])
    reward_grad_x_future, reward_grad_u = reward_grad(
        x_traj[0], x_traj[1:], u_traj, mdb_params, env_params
    )
    reward_grad_x = jnp.concatenate(
        [jnp.zeros((1, n_state)), reward_grad_x_future], axis=0
    )
    # reward_grad_x = jnp.zeros([horizon, n_state])
    # reward_grad_u = jnp.zeros([horizon, n_action])

    def get_logp_wo_x0(x0, x_traj_future, u_traj, mdb_params, env_params):
        x_traj = jnp.concatenate([x0[None], x_traj_future], axis=0)
        return get_logp_dynamics_scan(x_traj, u_traj, mdb_params, env_params)

    logp_dynamics_grad = jax.grad(get_logp_wo_x0, argnums=[1, 2])
    # get reward and logp_dynamics
    logp_dynamics_grad_x_future, logp_dynamics_grad_u = logp_dynamics_grad(
        x_traj[0], x_traj[1:], u_traj, mdb_params, env_params
    )
    logp_dynamics_grad_x = jnp.concatenate(
        [jnp.zeros((1, n_state)), logp_dynamics_grad_x_future], axis=0
    )

    # reward_scale = jnp.linalg.norm(logp_dynamics_grad_x) / jnp.linalg.norm(reward_grad_x)
    reward_scale = 10.0
    grad_x = logp_dynamics_grad_x + reward_grad_x * reward_scale

    # exit()
    # reward_scale = jnp.linalg.norm(logp_dynamics_grad_u) / jnp.linalg.norm(reward_grad_u)
    grad_u = logp_dynamics_grad_u + reward_grad_u * reward_scale
    # jax.debug.print('{x}', x = jnp.linalg.norm(reward_grad_u) * reward_scale)
    # jax.debug.print('{x}', x = jnp.linalg.norm(logp_dynamics_grad_u))

    # jax.debug.print(
    #     "dynamic likelihood gradient norm = {x}",
    #     x=jnp.linalg.norm(grad_x),
    # )
    # get new trajectory with Langevin dynamics
    eps = mdb_params.langevin_eps
    rng, rng_x, rng_u = jax.random.split(rng, 3)
    x_traj_new = (
        x_traj
        + eps * grad_x
        + jnp.sqrt(2 * eps) * jax.random.normal(rng_x, grad_x.shape)
    )
    x_traj_new = x_traj_new.at[0].set(
        env_params.init_state
    )  # NOTE: do not add noise to the initial state
    u_traj_new = (
        u_traj
        + eps * grad_u
        + jnp.sqrt(2 * eps) * jax.random.normal(rng_u, grad_u.shape)
    )

    return x_traj_new, u_traj_new


def plot_traj(
    x_traj: jnp.ndarray,
    u_traj: jnp.ndarray,
    x_traj_real: jnp.ndarray,
    filename: str = "traj",
):
    # create two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.quiver(
        x_traj[:, 0],
        x_traj[:, 1],
        -jnp.sin(x_traj[:, 2]),
        jnp.cos(x_traj[:, 2]),
        range(len(x_traj)),
        cmap="Reds",
    )
    ax1.plot(
        x_traj[:, 0],
        x_traj[:, 1],
        "r",
        alpha=0.2,
    )
    ax1.plot(
        x_traj_real[:, 0],
        x_traj_real[:, 1],
        "b--",
    )
    ax1.grid()
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_aspect("equal", adjustable="box")
    # plot star at [1, 0]
    ax1.plot(1.0, 0.0, "r*", markersize=16)
    # set title
    ax1.set_title("Trajectory")
    # plot circle at [0, 0]
    circle = plt.Circle((0, 0), 0.2, color="black", fill=False)
    ax1.add_artist(circle)
    # plot circle with dash line
    circle = plt.Circle((0, 0), 0.3, color="black", fill=False, linestyle="--")
    ax1.add_artist(circle)

    # plot x, y, theta
    # fig, ax2 = plt.subplots(1, 1)
    ax2.plot(x_traj[:, 0], "r", label="x")
    ax2.plot(x_traj[:, 1], "g", label="y")
    ax2.plot(x_traj[:, 2], "b", label="theta")
    ax2.plot(x_traj_real[:, 0], "r--", label="x_real")
    ax2.plot(x_traj_real[:, 1], "g--", label="y_real")
    ax2.plot(x_traj_real[:, 2], "b--", label="theta_real")
    ax2.grid()
    ax2.set_xlim([0, horizon])
    ax2.set_ylim([-1.5, 1.5])
    ax2.legend(loc="upper left")
    ax2.set_title("State")
    # plt.savefig(f"figure/{filename}_xytheta.png")
    # # release the plot
    # plt.close(fig)

    # plot T, tau
    # fig, ax3 = plt.subplots(1, 1)
    ax3.plot(u_traj[:, 0], "c", label="$T$")
    ax3.plot(u_traj[:, 1], "m", label="$tau$")
    ax3.grid()
    ax3.set_xlim([0, horizon])
    ax3.set_ylim([-2.0, 2.0])
    ax3.legend(loc="upper left")
    ax3.set_title("Control")
    # plt.savefig(f"figure/{filename}_u.png")
    # # release the plot
    # plt.close(fig)

    plt.savefig(f"figure/{filename}.png")
    plt.savefig(f"figure/traj.png")
    # release the plot
    plt.close(fig)


def main():
    # check NaN with jax
    # jax.config.update("jax_debug_nans", True)
    rng = jax.random.PRNGKey(0)

    # schedule noise here
    noise_std_init = 0.2
    noise_std_final = 5e-3
    diffuse_step = 10
    diffuse_substeps = 20
    # noise_std_schedule = jnp.ones(diffuse_step) * noise_std_final
    langevin_eps_schedule = jnp.linspace(1.0, 1e-1, diffuse_step) * 1e-5  # 1e-5
    # plan in exponential space
    noise_std_schedule = jnp.exp(
        jnp.linspace(jnp.log(noise_std_init), jnp.log(noise_std_final), diffuse_step)
    )
    # noise_var_schedule = noise_std_schedule**2
    # noise_var_diff = -jnp.diff(noise_var_schedule, append=0.0)
    # langevin_eps_schedule = jnp.sqrt(noise_var_diff/diffuse_substeps)

    # init env and mbd params
    env_params = EnvParams()
    mdb_params = MBDParams()

    # init trajectory
    rng, rng_x, rng_u = jax.random.split(rng, 3)
    x_traj = jax.random.normal(rng_x, (horizon, n_state))
    u_traj = jax.random.normal(rng_u, (horizon, n_action))
    x_traj = x_traj.at[0].set(env_params.init_state)

    # test kalman filter
    # # generate feasible initial trajectory
    # x_traj_real = jnp.zeros((env_params.horizon, env_params.n_state))
    # u_traj_real = jax.random.normal(rng_u, (env_params.horizon, env_params.n_action))
    # x_traj_real = x_traj_real.at[0].set(env_params.init_state)
    # for t in range(1, env_params.horizon):
    #     x_traj_real = x_traj_real.at[t].set(
    #         get_A(x_traj_real[t-1], env_params) @ x_traj_real[t-1]
    #         + get_B(x_traj_real[t-1], env_params) @ u_traj_real[t-1]
    #     )
    # # plot the trajectory
    # plot_traj(x_traj_real, "init_traj")

    # # add noise to the initial trajectory
    # x_traj = x_traj_real + jax.random.normal(rng_x, x_traj_real.shape) * 1.0
    # x_traj = x_traj.at[0].set(env_params.init_state)
    # u_traj = u_traj_real + jax.random.normal(rng_u, u_traj_real.shape) * 1.0
    # # use kalman filter to estimate the initial state
    # mdb_params = mdb_params.replace(
    #     noise_std=1.0,
    # )
    # logp_dynamics, x_traj_filtered = get_logp_dynamics(x_traj, u_traj, mdb_params, env_params)
    # # plot the trajectory
    # plot_traj(x_traj_filtered, "init_traj_filtered")
    # plot_traj(x_traj, "init_traj_noisy")
    # jax.debug.print('logp_dynamics = {x}', x=logp_dynamics)
    # jax.debug.print('grad logp_dynamics = {x}', x=jax.grad(get_logp_dynamics_scan, argnums=[0])(x_traj, u_traj, mdb_params, env_params))
    # exit()

    # run MBD
    x_traj_save = []
    get_next_traj_jit = jax.jit(get_next_traj)
    for d_step in range(diffuse_step):
        mdb_params = mdb_params.replace(
            noise_std=noise_std_schedule[d_step],
        )
        for substep in range(diffuse_substeps):
            mdb_params = mdb_params.replace(
                langevin_eps=langevin_eps_schedule[substep]
                * (mdb_params.noise_std / 5e-3) ** 2
            )
            rng, rng_traj = jax.random.split(rng)
            x_traj, u_traj = get_next_traj_jit(
                x_traj, u_traj, mdb_params, env_params, rng_traj
            )
            logp_dynamics = get_logp_dynamics_scan(
                x_traj, u_traj, mdb_params, env_params
            )
            logp_reward = get_reward(x_traj, u_traj, mdb_params, env_params)
            jax.debug.print(
                "d_step = {d_step}, substep = {substep}, logp_dynamic = {x}, logp_reward = {y}",
                d_step=d_step,
                substep=substep,
                x=logp_dynamics,
                y=logp_reward,
            )
            # rollout dynamics to get real trajectory
            x_traj_real = jnp.zeros((horizon, n_state))
            x_traj_real = x_traj_real.at[0].set(env_params.init_state)
            for t in range(1, horizon):
                x_traj_real = x_traj_real.at[t].set(
                    get_A(x_traj_real[t - 1], env_params) @ x_traj_real[t - 1]
                    + get_B(x_traj_real[t - 1], env_params) @ u_traj[t - 1]
                )
            plot_traj(
                x_traj, u_traj, x_traj_real, f"traj_{d_step*diffuse_substeps+substep}"
            )
        # save trajectory
        x_traj_save.append(x_traj)

    # save data
    jnp.savez("diffuse_traj.npz", x_traj_save=jnp.stack(x_traj_save))


if __name__ == "__main__":
    main()
