from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp
import pickle
from functools import partial
from matplotlib import pyplot as plt

import mbd


class HumanoidTrack(PipelineEnv):

    def __init__(self, mode="jog"):
        sys = mjcf.load(f"{mbd.__path__[0]}/assets/humanoidtrack.xml")
        self.H = 50  # traj time 1.5s
        body_names = [
            "torso",
            "left_thigh",
            "right_thigh",
            "left_shin",
            "right_shin",
        ]
        self.track_body_names = body_names
        self.track_body_idx = jnp.array(
            [sys.link_names.index(name) for name in self.track_body_names]
        )
        self.ref_body_names = [name + "_ref" for name in body_names]
        self.ref_body_idx = jnp.array(
            [sys.link_names.index(name) for name in self.ref_body_names]
        )
        with open(f"{mbd.__path__[0]}/assets/jog_xref.pkl", "rb") as f:
            xs_demo_dict = pickle.load(f)
        xref = []
        for name in body_names:
            x = xs_demo_dict[name]
            if len(x) < self.H:
                x = jnp.concatenate([x, jnp.tile(x[-1:], (self.H - len(x), 1))], axis=0)
            else:
                x = x[70 : (self.H + 70)]
            xref.append(x)
        self.xref = jnp.stack(xref, axis=0)
        self.rew_xref = 1.0

        super().__init__(sys=sys, backend="positional", n_frames=5)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        qpos = self.sys.init_q
        qvel = jnp.zeros(self.sys.qd_size())

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_linup": zero,
            "reward_quadctrl": zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        # set reference state for visualization
        for i, idx in enumerate(self.ref_body_idx):
            pipeline_state = pipeline_state.replace(
                x=pipeline_state.x.replace(
                    pos=pipeline_state.x.pos.at[idx].set(
                        self.xref[i, jnp.int32(state.done)]
                    ),
                )
            )
        # quad_impact_cost is not computed here

        obs = self._get_obs(pipeline_state)
        reward = self._get_reward(state)

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=state.done + 1
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd], axis=-1)

    def _get_reward(self, state) -> jax.Array:
        # x_feet = state.pipeline_state.x.pos[self.track_body_idx[-2:], 0]
        # x_feet_ref = self.xref[-2:, jnp.int32(state.done), 0]
        # err_feet = jnp.abs(x_feet - x_feet_ref)
        # err_feet = jnp.clip(err_feet, 0.0, 0.5)
        return 1.0 + (
            -jnp.abs(state.pipeline_state.xd.vel[0, 0] - 1.6)
            - jnp.abs(state.pipeline_state.x.pos[0, 2] - 1.3)
            - jnp.abs(state.pipeline_state.x.pos[0, 1]) * 0.1
        )

    @partial(jax.jit, static_argnums=(0,))
    def eval_xref_logpd(self, pipeline_state):
        xs = pipeline_state.x.pos[:, self.track_body_idx].transpose(1, 0, 2)
        xs_err = xs - self.xref
        logpd = (
            0.0
            - ((jnp.clip(jnp.linalg.norm(xs_err, axis=-1), 0.0, 0.5) / 0.5) ** 2).mean()
        )
        return logpd
