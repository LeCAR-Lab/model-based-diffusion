import jax
from jax import numpy as jnp
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
from flax import struct
from functools import partial


@struct.dataclass
class State:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


def dynamics(x, u):
    return jnp.array(
        [
            x[3] * jnp.sin(x[2]),  # x_dot
            x[3] * jnp.cos(x[2]),  # y_dot
            x[3] * u[0] * jnp.pi,  # theta_dot
            u[1],  # v_dot
        ]
    )

def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Car:
    def __init__(self):
        obs_radius = 0.3
        self.obs_radius = obs_radius
        self.obs_center = jnp.array(
            [
                [-obs_radius * 3, obs_radius * 2],
                [-obs_radius * 2, obs_radius * 2],
                [-obs_radius * 1, obs_radius * 2],
                [0.0, obs_radius * 2],
                [0.0, obs_radius * 1],
                [0.0, 0.0],
                [0.0, -obs_radius * 1],
                [-obs_radius * 3, -obs_radius * 2],
                [-obs_radius * 2, -obs_radius * 2],
                [-obs_radius * 1, -obs_radius * 2],
                [0.0, -obs_radius * 2],
            ]
        )
        self.x_init = jnp.array([-0.5, 0.0, jnp.pi * 0.5, 0.0])
        self.x_goal = jnp.array([0.5, 0.0, jnp.pi * 0.5, 0.0])

    @partial(jax.jit, static_argnums=0)
    def reset(self, rng: jnp.ndarray) -> State:
        reward = (
            1.0
            - jnp.clip(jnp.linalg.norm(self.x_init[:2] - self.x_goal[:2]), 0.0, 0.5)
            * 2.0
        )
        done = 0.0
        return State(obs=self.x_init, reward=reward, done=done)

    # @partial(jax.jit, static_argnums=0)
    def step(self, state: State, action: jnp.ndarray) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        x = state.obs
        x_new = rk4(dynamics, x, action, 0.3)
        pos_new = x_new[:2]
        hit_wall = jnp.any(
            jnp.linalg.norm(pos_new[None] - self.obs_center, axis=-1) < self.obs_radius
        )
        x_new = jnp.where(hit_wall, state.obs, x_new)
        reward = (
            1.0 - jnp.clip(jnp.linalg.norm(x_new[:2] - self.x_goal[:2]), 0.0, 0.5) * 2.0
        )
        done = 0.0
        return state.replace(obs=x_new, reward=reward, done=done)

    @property
    def action_size(self):
        return 2

    @property
    def observation_size(self):
        return 4


def vis_env(ax, Y0s):
    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    obs_radius = 0.3
    obs_center = jnp.array(
        [
            [-obs_radius * 3, obs_radius * 2],
            [-obs_radius * 2, obs_radius * 2],
            [-obs_radius * 1, obs_radius * 2],
            [0.0, obs_radius * 2],
            [0.0, obs_radius * 1],
            [0.0, 0.0],
            [0.0, -obs_radius * 1],
            [-obs_radius * 3, -obs_radius * 2],
            [-obs_radius * 2, -obs_radius * 2],
            [-obs_radius * 1, -obs_radius * 2],
            [0.0, -obs_radius * 2],
        ]
    )
    for i in range(obs_center.shape[0]):
        circle = Circle(obs_center[i], obs_radius, color="k", fill=True, alpha=0.5)
        ax.add_artist(circle)
    N = Y0s.shape[1]
    for i in range(Y0s.shape[0]):
        x = Y0s[i].T
        ax.plot(x[0, :], x[1, :], "b-", alpha=0.2)
        ax.quiver(
            x[0, :], x[1, :], jnp.sin(x[2, :]), jnp.cos(x[2, :]), range(N), cmap="Blues"
        )


def main():
    rng = jax.random.PRNGKey(0)
    rng, rng_reset = jax.random.split(rng)
    env = Car()
    state = env.reset(rng=rng_reset)
    poses = []
    for _ in range(20):
        rng, rng_action = jax.random.split(rng)
        action = jnp.clip(jax.random.normal(rng_action, (2,)), -1.0, 1.0)
        state = env.step(state, action)
        poses.append(state.obs)
    Y0s = jnp.stack(poses)[None]
    fig, ax = plt.subplots()
    vis_env(ax, Y0s)
    plt.savefig("../figure/point.png")


if __name__ == "__main__":
    main()
