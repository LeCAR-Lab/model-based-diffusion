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

class Point:
    def __init__(self):
        obs_radius = 0.3
        self.obs_radius = obs_radius
        self.obs_center = jnp.array([
            [-obs_radius * 3, obs_radius * 2], [-obs_radius * 2, obs_radius * 2], [-obs_radius * 1, obs_radius * 2], [0.0, obs_radius * 2],
            [0.0, obs_radius * 1], [0.0, 0.0], [0.0, -obs_radius * 1],
            [-obs_radius * 3, -obs_radius * 2], [-obs_radius * 2, -obs_radius * 2], [-obs_radius * 1, -obs_radius * 2], [0.0, -obs_radius * 2],
        ])
        self.x_init = jnp.array([-0.5, 0.0])
        self.x_goal = jnp.array([0.5, 0.0])


    @partial(jax.jit, static_argnums=0)
    def reset(self, rng: jnp.ndarray) -> State:
        pos = self.x_init
        reward = 1.0 - jnp.clip(jnp.linalg.norm(pos - self.x_goal), 0.0, 0.5) * 2.0
        done = 0.0
        return State(obs=pos, reward=reward, done=done)

    @partial(jax.jit, static_argnums=0)
    def step(self, state: State, action: jnp.ndarray) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        pos_new = state.obs + 0.2 * action
        hit_wall = jnp.any(jnp.linalg.norm(pos_new[None] - self.obs_center, axis=-1) < self.obs_radius)
        pos_new = jnp.where(hit_wall, state.obs, pos_new)
        reward = 1.0 - jnp.clip(jnp.linalg.norm(pos_new - self.x_goal), 0.0, 0.5) * 2.0
        done = 0.0
        return state.replace(obs=pos_new, reward=reward, done=done)

    @property
    def action_size(self):
        return 2

    @property
    def observation_size(self):
        return 2
    
def vis_env(ax, Y0s):
    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    obs_radius = 0.3
    obs_center = jnp.array([
        [-obs_radius * 3, obs_radius * 2], [-obs_radius * 2, obs_radius * 2], [-obs_radius * 1, obs_radius * 2], [0.0, obs_radius * 2],
        [0.0, obs_radius * 1], [0.0, 0.0], [0.0, -obs_radius * 1],
        [-obs_radius * 3, -obs_radius * 2], [-obs_radius * 2, -obs_radius * 2], [-obs_radius * 1, -obs_radius * 2], [0.0, -obs_radius * 2],
    ])
    for i in range(obs_center.shape[0]):
        circle = Circle(obs_center[i], obs_radius, color="k", fill=True, alpha=0.5)
        ax.add_artist(circle)
    for i in range(Y0s.shape[0]):
        ax.plot(Y0s[i, :, 0], Y0s[i, :, 1], "-o", color="blue")

def main():
    rng = jax.random.PRNGKey(0)
    rng, rng_reset = jax.random.split(rng)
    env = Point()
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
    plt.show()
    plt.savefig("../figure/point.png")

if __name__ == "__main__":
    main()