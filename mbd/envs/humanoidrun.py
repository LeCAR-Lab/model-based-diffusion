from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jnp

import mbd


class HumanoidRun(PipelineEnv):

    def __init__(self):
        sys = mjcf.load(f"{mbd.__path__[0]}/assets/humanoidrun.xml")

        super().__init__(sys=sys, backend="positional", n_frames=7)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -0.01, 0.01
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01
        )
        qvel = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jnp.zeros(self.sys.act_size()))
        reward, done = jnp.zeros(2)
        return State(pipeline_state, obs, reward, done, {})

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        obs = self._get_obs(pipeline_state, action)
        reward = self._get_reward(pipeline_state)

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd], axis=-1)

    def _get_reward(self, pipeline_state: base.State) -> jax.Array:
        return (
            pipeline_state.x.pos[0, 0] * 1.0
            - jnp.clip(jnp.abs(pipeline_state.x.pos[0, 2] - 1.25), -1.0, 1.0) * 1.0
            - jnp.abs(pipeline_state.x.pos[0, 1]) * 0.1
        )
