from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp

import mbd


class Cartpole(PipelineEnv):
    def __init__(self, backend="positional", **kwargs):
        sys = mjcf.load(f"{mbd.__path__[0]}/assets/cartpole.xml")

        n_frames = 2

        if backend in ["spring", "positional"]:
            sys = sys.replace(dt=0.005)
            n_frames = 4

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01
        ) + jp.array([0.0, jp.pi])
        qd = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=-0.01, maxval=0.01)
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        reward = jp.cos(pipeline_state.q[1]) - jp.abs(pipeline_state.qd[0])
        done = 0.0
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    @property
    def action_size(self):
        return 1

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe cartpole body position and velocities."""
        return jp.concatenate([pipeline_state.q, pipeline_state.qd])
