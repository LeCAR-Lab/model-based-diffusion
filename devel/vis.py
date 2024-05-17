import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
from flax import struct

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


def plot_reward():
    # default parameters
    params = Params()
    # generate x, y grid from -1.5 to 1.5
    x = jnp.linspace(-1.5, 1.5, 100)
    y = jnp.linspace(-1.5, 1.5, 100)
    X, Y = jnp.meshgrid(x, y)

    # generate reward
    def get_single_point_reward(x, y):
        x_traj = jnp.array([x, y, 0.0, 0.0])[None]
        u_traj = jnp.array([0.0, 0.0])[None]
        return get_reward(x_traj, u_traj, params)

    get_single_point_reward = jax.jit(get_single_point_reward)
    Z = jax.vmap(jax.vmap(get_single_point_reward))(X, Y)
    # plot the reward, plot with red color
    plt.contourf(X, Y, Z, levels=20, cmap="Reds")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Reward")
    plt.savefig("figure/reward.png")
    plt.close()


def plot_traj(
    x_traj: jnp.ndarray,
    u_traj: jnp.ndarray,
    x_traj_real: jnp.ndarray,
    log_info: dict,
    filename: str = "",
):
    # create 2x4 subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))

    ax = axes[0, 0]
    # color each point with cmap red
    ax.scatter(
        x_traj[:, 0],
        x_traj[:, 1],
        c=range(horizon),
        cmap="Reds",
        marker="o",
        alpha=1.0,
    )
    ax.plot(
        x_traj[:, 0],
        x_traj[:, 1],
        "r",
        alpha=0.2,
    )
    ax.plot(
        x_traj_real[:, 0],
        x_traj_real[:, 1],
        "b--",
    )
    ax.grid()
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_aspect("equal", adjustable="box")
    # plot star at [1, 0]
    ax.plot(1.0, 0.0, "r*", markersize=16)
    # set title
    ax.set_title("Trajectory")
    # # plot circle at [0, 0]
    # circle = plt.Circle((0, 0), 0.2, color="black", fill=False)
    # ax.add_artist(circle)
    # # plot circle with dash line
    # circle = plt.Circle((0, 0), 0.3, color="black", fill=False, linestyle="--")
    # ax.add_artist(circle)

    # plot x, y, theta
    # fig, ax = plt.subplots(1, 1)
    ax = axes[0, 1]
    ax.plot(x_traj[:, 0], "r", label="x")
    ax.plot(x_traj[:, 1], "g", label="y")
    # ax.plot(x_traj[:, 2], "b", label="theta")
    ax.plot(x_traj_real[:, 0], "r--", label="x_real")
    ax.plot(x_traj_real[:, 1], "g--", label="y_real")
    # ax.plot(x_traj_real[:, 2], "b--", label="theta_real")
    ax.grid()
    ax.set_xlim([0, horizon])
    ax.set_ylim([-1.5, 1.5])
    ax.legend(loc="upper left")
    ax.set_title("State")
    # plt.savefig(f"figure/{filename}_xytheta.png")
    # # release the plot
    # plt.close(fig)

    # plot T, tau
    # fig, ax = plt.subplots(1, 1)
    ax = axes[0, 2]
    ax.plot(u_traj[:, 0], "c", label="$T$")
    ax.plot(u_traj[:, 1], "m", label="$tau$")
    ax.grid()
    ax.set_xlim([0, horizon])
    ax.set_ylim([-2.0, 2.0])
    ax.legend(loc="upper left")
    ax.set_title("Control")
    # plt.savefig(f"figure/{filename}_u.png")
    # # release the plot
    # plt.close(fig)

    ax = axes[1, 0]
    ax.plot(log_info["logp_dynamics"], "r", label="logp_dynamics")
    for i in range(diffuse_step):
        ax.axvline(x=i * diffuse_substeps, color="black", linestyle="--")
    ax.grid()
    ax.legend(loc="upper left")
    ax.set_title("Dynamic Log Probability")

    ax = axes[1, 1]
    ax.plot(log_info["logp_reward"], "b", label="logp_reward")
    for i in range(diffuse_step):
        ax.axvline(x=i * diffuse_substeps, color="black", linestyle="--")
    ax.grid()
    ax.legend(loc="upper left")
    ax.set_title("Reward")

    ax = axes[1, 2]
    ax.plot(log_info["noise_std"], "black", label="noise_std")
    ax.grid()
    ax.legend(loc="upper left")
    ax.set_title("Diffusion Noise Std")

    ax = axes[1, 3]
    ax.plot(log_info["langevin_eps"], "black", label="langevin_eps")
    ax.grid()
    ax.legend(loc="upper left")
    ax.set_title("Langevin Eps")

    if filename == "":
        plt.show()
    else:
        plt.savefig(f"figure/{filename}.png")
        plt.savefig(f"figure/traj.png")
        # release the plot
        plt.close(fig)
