from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, html
from etils import epath
import jax
from jax import numpy as jnp
import mujoco


class HumanoidTrack(PipelineEnv):

    def __init__(self, backend="positional", **kwargs):
        path = "humanoidtrack.xml"
        sys = mjcf.load(path)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.replace(dt=0.006)
            n_frames = 5
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=jnp.array(
                        [
                            350.0,
                            350.0,
                            350.0,
                            350.0,
                            350.0,
                            350.0,
                            350.0,
                            350.0,
                            350.0,
                            350.0,
                            350.0,
                            100.0,
                            100.0,
                            100.0,
                            100.0,
                            100.0,
                            100.0,
                        ]
                    )
                )
            )  # pyformat: disable

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

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

        # quad_impact_cost is not computed here

        obs = self._get_obs(pipeline_state)
        reward = self._get_reward(pipeline_state)

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd], axis=-1)

    def _get_reward(self, pipeline_state: base.State) -> jax.Array:
        return pipeline_state.x.pos[0, 0] - jnp.abs(pipeline_state.x.pos[0, 2] - 1.2)


def main():
    env = HumanoidTrack()
    rng = jax.random.PRNGKey(1)
    env_step = jax.jit(env.step)
    env_reset = jax.jit(env.reset)
    state = env_reset(rng)
    rollout = [state.pipeline_state]
    for _ in range(50):
        rng, rng_act = jax.random.split(rng)
        act = jax.random.uniform(rng_act, (env.action_size,), minval=-1.0, maxval=1.0)
        state = env_step(state, act)
        print(state.pipeline_state.x.pos[0, 2])
        rollout.append(state.pipeline_state)
    webpage = html.render(env.sys.replace(dt=env.dt), rollout)
    with open("../figure/humanoid.html", "w") as f:
        f.write(webpage)


if __name__ == "__main__":
    main()
