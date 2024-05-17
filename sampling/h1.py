import jax
from jax import numpy as jnp
import brax
from brax.envs.base import PipelineEnv, State
from brax.positional import pipeline
from matplotlib.patches import Circle, Rectangle
from matplotlib import transforms
import matplotlib.pyplot as plt
from brax.io import html

from brax.io import mjcf


class PushT(PipelineEnv):
    def __init__(self, backend: str = "positional"):
        sys = mjcf.load("robots/h1_pos.xml")

        super().__init__(sys, backend=backend, n_frames=5)

    def reset(self, rng: jnp.ndarray) -> State:
        q = self.sys.init_q
        qd = jnp.zeros(self.sys.qd_size())
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward = self._get_reward(pipeline_state)
        done = self._get_done(pipeline_state)
        metrics = {}
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        reward = self._get_reward(pipeline_state)
        done = self._get_done(pipeline_state)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: pipeline.State) -> jnp.ndarray:
        return jnp.concat([pipeline_state.q, pipeline_state.qd], axis=-1)

    def _get_reward(self, pipeline_state: pipeline.State) -> jnp.ndarray:
        return 0.0

    def _get_done(self, pipeline_state: pipeline.State) -> jnp.ndarray:
        return 0.0

    @property
    def action_size(self):
        return 2


def main():
    env = PushT()
    rng = jax.random.PRNGKey(1)
    env_step = jax.jit(env.step)
    env_reset = jax.jit(env.reset)
    state = env_reset(rng)
    rollout = [state.pipeline_state]
    for _ in range(50):
        rng, rng_act = jax.random.split(rng)
        act = jax.random.uniform(rng_act, (env.action_size,), minval=-1.0, maxval=1.0)
        state = env_step(state, act)
        rollout.append(state.pipeline_state)
    webpage = html.render(env.sys.replace(dt=env.dt), rollout)
    with open("../figure/h1.html", "w") as f:
        f.write(webpage)


if __name__ == "__main__":
    main()
