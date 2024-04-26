import jax
from jax import numpy as jnp
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
from flax import struct

@struct.dataclass
class State:
  obs: jax.Array
  reward: jax.Array
  done: jax.Array

class Point:
    def __init__(self):
        pass

    def reset(self, rng: jnp.ndarray) -> State:
        pos = jnp.array([-1.0, 0.0])
        reward = 1.0 - jnp.linalg.norm(pos - jnp.array([1.0, 0.0])) / 2.0
        done = 0.0
        return State(obs=pos, reward=reward, done=done)

    def step(self, state: State, action: jnp.ndarray) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        pos_new = state.obs + 0.2 * action
        hit_wall = jnp.linalg.norm(pos_new, ord=jnp.inf) < 0.5
        pos_new = jnp.where(hit_wall, state.obs, pos_new)
        reward = 1.0 - jnp.linalg.norm(pos_new - jnp.array([1.0, 0.0])) / 2.0
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
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, color="black"))
    for i in range(Y0s.shape[0]):
        ax.plot(*Y0s[i].T, "-o", color="blue")

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
    plt.savefig("../figure/point.png")

if __name__ == "__main__":
    main()