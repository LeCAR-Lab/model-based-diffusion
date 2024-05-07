from typing import Tuple

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp


class Hopper(PipelineEnv):
    def __init__(self):
        path = epath.resource_path("brax") / "envs/assets/hopper.xml"
        sys = mjcf.load(path)

        self._reset_noise_scale = 5e-3

        super().__init__(sys=sys, backend="positional", n_frames=20)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        return State(pipeline_state, obs, reward, done, {})

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        obs = self._get_obs(pipeline_state)
        reward = self._get_reward(pipeline_state)

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=0.0
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        position = pipeline_state.q
        position = position.at[1].set(pipeline_state.x.pos[0, 2])
        velocity = jp.clip(pipeline_state.qd, -10, 10)

        return jp.concatenate((position, velocity))

    def _get_reward(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment reward."""
        return (
            pipeline_state.x.pos[0, 0]
            - jp.clip(jp.abs(pipeline_state.x.pos[0, 2] - 1.2), -1.0, 1.0)
        )
