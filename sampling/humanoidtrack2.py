from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, html
from etils import epath
import jax
from jax import numpy as jnp
import pickle


class HumanoidTrack(PipelineEnv):

    def __init__(self, backend="positional", **kwargs):
        path = "humanoidtrack2.xml"
        sys = mjcf.load(path)
        n_frames = 5
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        self.torso_idx = sys.link_names.index("torso")
        self.left_shin_idx = sys.link_names.index("left_shin")
        self.right_shin_idx = sys.link_names.index("right_shin")

        body_names = [
            "pelvis",
            "head",
            "ltoe",
            "rtoe",
            "lheel",
            "rheel",
            "lknee",
            "rknee",
            "lhand",
            "rhand",
            "lelbow",
            "relbow",
            "lshoulder",
            "rshoulder",
            "lhip",
            "rhip",
            "torso",
            "left_thigh",
            "right_thigh",
            "left_shin",
            "right_shin",
        ]
        self.ref_body_names = [name + "_ref" for name in body_names]
        self.ref_body_idx = {
            name: sys.link_names.index(name) for name in self.ref_body_names
        }

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
        return (
            pipeline_state.x.pos[0, 0]
            - jnp.clip(jnp.abs(pipeline_state.x.pos[0, 2] - 1.25), -1.0, 1.0)
            - jnp.abs(pipeline_state.x.pos[0, 1]) * 0.1
        )


def set_ref_body_pos(env, pipeline_state, xs_ref_dict, t):
    for i, (name, idx) in enumerate(env.ref_body_idx.items()):
        pipeline_state = pipeline_state.replace(
            x=pipeline_state.x.replace(
                pos=pipeline_state.x.pos.at[idx].set(xs_ref_dict[name][t])
            )
        )
    return pipeline_state


def main():
    env = HumanoidTrack()

    task = "run"
    xs_ref = jnp.load(f"../devel/{task}_ref.npy")
    if task == "walk":
        X = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    elif task == "run":
        X = jnp.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        raise ValueError(f"task {task} not recognized")
    xs_ref = jnp.einsum("ij,kpj->kpi", X, xs_ref)
    # modify xs_ref
    if task == "walk":
        xs_ref = xs_ref[50:]  # remove first 50 frames
    elif task == "run":
        xs_ref = jnp.concatenate(
            [xs_ref[:1]] * 8 + [xs_ref[1:]], axis=0
        )  # repeat first frame 8 times
        # scale to 0.75
        ts = jnp.arange(xs_ref.shape[0]) 
        ts_scaled = ts * 0.75
        xs_ref = jax.vmap(
            jax.vmap(jnp.interp, in_axes=(None, None, 1)), in_axes=(None, None, 1)
        )(ts_scaled, ts, xs_ref)
        xs_ref = jnp.moveaxis(xs_ref, 2, 0)
        print(xs_ref.shape)
    xs_ref_dict = {name: xs_ref[:, i] for i, name in enumerate(env.ref_body_names)}
    xs_ref_dict["torso_ref"] = 0.5 * (
        xs_ref_dict["head_ref"] + xs_ref_dict["pelvis_ref"]
    )
    xs_ref_dict["left_thigh_ref"] = 0.5 * (
        xs_ref_dict["lhip_ref"] + xs_ref_dict["lknee_ref"]
    )
    xs_ref_dict["right_thigh_ref"] = 0.5 * (
        xs_ref_dict["rhip_ref"] + xs_ref_dict["rknee_ref"]
    )
    xs_ref_dict["left_shin_ref"] = 0.5 * (
        xs_ref_dict["lknee_ref"] + xs_ref_dict["ltoe_ref"]
    )
    xs_ref_dict["right_shin_ref"] = 0.5 * (
        xs_ref_dict["rknee_ref"] + xs_ref_dict["rtoe_ref"]
    )

    xs_ref_dict_rm_ref = {name[:-4]: xs_ref_dict[name] for name in xs_ref_dict}
    # save xs_ref_dict
    with open(f"../devel/{task}_ref_dict.pkl", "wb") as f:
        pickle.dump(xs_ref_dict_rm_ref, f)

    rng = jax.random.PRNGKey(1)
    env_step = jax.jit(env.step)
    env_reset = jax.jit(env.reset)
    state = env_reset(rng)
    rollout = [state.pipeline_state]
    for t in range(200):
        rng, rng_act = jax.random.split(rng)
        act = jax.random.uniform(rng_act, (env.action_size,), minval=-1.0, maxval=1.0)
        state = env_step(state, act)
        pipeline_state = state.pipeline_state
        pipeline_state = set_ref_body_pos(env, pipeline_state, xs_ref_dict, t)
        rollout.append(pipeline_state)
    webpage = html.render(env.sys.replace(dt=env.dt), rollout)
    with open("../figure/humanoid.html", "w") as f:
        f.write(webpage)


if __name__ == "__main__":
    main()
