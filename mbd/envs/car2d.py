import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt

import mbd


def car_dynamics(x, u):
    # x = x.at[3].set(jnp.clip(x[3], -2.0, 2.0))
    return jnp.array(
        [
            u[1] * jnp.sin(x[2])*3.0,  # x_dot
            u[1] * jnp.cos(x[2])*3.0,  # y_dot
            u[0] * jnp.pi / 3 * 2.0,  # theta_dot
            # u[1] * 6.0,  # v_dot
        ]
    )


def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def check_collision(x, obs_center, obs_radius):
    dist2objs = jnp.linalg.norm(x[:2] - obs_center, axis=1)
    return jnp.any(dist2objs < obs_radius)


@struct.dataclass
class State:
    pipeline_state: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


class Car2d:
    def __init__(self):
        self.dt = 0.1
        self.H = 50
        r_obs = 0.3
        self.obs_center = jnp.array(
            [
                # [-r_obs * 3, r_obs * 5],
                # [-r_obs * 3, r_obs * 4],
                # [-r_obs * 3, r_obs * 3],
                [-r_obs * 3, r_obs * 2],
                [-r_obs * 2, r_obs * 2],
                [-r_obs * 1, r_obs * 2],
                [0.0, r_obs * 2],
                [0.0, r_obs * 1],
                [0.0, 0.0],
                [0.0, -r_obs * 1],
                [-r_obs * 3, -r_obs * 2],
                # [-r_obs * 3, -r_obs * 3],
                # [-r_obs * 3, -r_obs * 4],
                # [-r_obs * 3, -r_obs * 5],
                [-r_obs * 2, -r_obs * 2],
                [-r_obs * 1, -r_obs * 2],
                [0.0, -r_obs * 2],
            ]
        )
        self.obs_radius = r_obs  # Radius of the obstacle
        self.x0 = jnp.array([-0.5, 0.0, jnp.pi*3/2])
        self.xg = jnp.array([0.5, 0.0, 0.0])
        self.xref = jnp.load(f"{mbd.__path__[0]}/assets/car2d_xref.npy")
        xref_diff = jnp.diff(self.xref, axis=0)
        theta = jnp.arctan2(xref_diff[:, 0], xref_diff[:, 1])
        self.thetaref = jnp.append(theta, theta[-1])
        self.rew_xref = jax.vmap(self.get_reward)(self.xref).mean()

    def reset(self, rng: jax.Array):
        """Resets the environment to an initial state."""
        return State(self.x0, self.x0, 0.0, 0.0)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        action = jnp.clip(action, -1.0, 1.0)
        q = state.pipeline_state
        q_new = rk4(car_dynamics, state.pipeline_state, action, self.dt)
        collide = check_collision(q_new, self.obs_center, self.obs_radius)
        q = jnp.where(collide, q, q_new)
        reward = self.get_reward(q)
        return state.replace(pipeline_state=q, obs=q, reward=reward, done=0.0)

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, q):
        reward = (
            1.0 - (jnp.clip(jnp.linalg.norm(q[:2] - self.xg[:2]), 0.0, 0.2) / 0.2) ** 2
        )
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def eval_xref_logpd(self, xs):
        xs_err = xs[:, :2] - self.xref[:, :2]
        # theta_err = xs[:, 3] - self.thetaref
        logpd = 0.0-(
            (jnp.clip(jnp.linalg.norm(xs_err, axis=-1), 0.0, 0.5) / 0.5) ** 2
        ).mean(axis=-1)
        return logpd

    @property
    def action_size(self):
        return 2

    @property
    def observation_size(self):
        return 3

    def render(self, ax, xs: jnp.ndarray):
        # obstacles
        for i in range(self.obs_center.shape[0]):
            circle = plt.Circle(
                self.obs_center[i, :], self.obs_radius, color="k", fill=True, alpha=0.5
            )
            ax.add_artist(circle)
        ax.quiver(
            xs[:, 0],
            xs[:, 1],
            jnp.sin(xs[:, 2]),
            jnp.cos(xs[:, 2]),
            range(self.H + 1),
            cmap="Reds",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_title("Car 2D")
