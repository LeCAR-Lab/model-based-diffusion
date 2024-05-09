import jax
from jax import numpy as jnp
from brax.io import html
from brax.io.html import render_from_json
from brax.io.json import _to_dict, _GEOM_TYPE_NAMES
import json
import numpy as np
from jax.tree_util import tree_map


import mbd

env_name = "pushT"
env = mbd.envs.get_env(env_name)
step_env_jit = jax.jit(env.step)
Hsample = 50
path = f"{mbd.__path__[0]}/../results/{env_name}"
mu_0ts = jnp.load(f"{path}/mu_0ts.npy")

def dumps(sys, statess) -> str:

    d = _to_dict(sys)

    # Fill in empty link names
    link_names = [n or f'link {i}' for i, n in enumerate(sys.link_names)]
    link_names += ['world']

    # Unpack geoms into a dict for the visualizer
    link_geoms = {}
    for id_ in range(sys.ngeom):
        link_idx = sys.geom_bodyid[id_] - 1
        rgba = sys.geom_rgba[id_]
        # if (rgba == [0.5, 0.5, 0.5, 1.0]).all():
        #     rgba = np.array([0.4, 0.33, 0.26, 1.0])

        geom = {
            'name': _GEOM_TYPE_NAMES[sys.geom_type[id_]],
            'link_idx': link_idx,
            'pos': sys.geom_pos[id_],
            'rot': sys.geom_quat[id_],
            'rgba': rgba,
            'size': sys.geom_size[id_],
        }

        link_geoms.setdefault(link_names[link_idx], []).append(_to_dict(geom))

    # print(len(link_names))
    # repeat link_geoms for each body across all timesteps
    all_link_geoms = {}
    all_link_names = []
    traj_len = len(statess[0])
    for k in range(traj_len):
        for i, (name, geoms) in enumerate(link_geoms.items()):
            name = f'{name}_{k}' if k > 0 else name
            geoms_new = []
            for geom in geoms:
                geom_new = geom.copy()
                if 'world' in name:
                    geom_new['link_idx'] = -1
                else:
                    geom_new['link_idx'] = geom['link_idx'] + k * (len(link_names)-1)
                    a = k / traj_len * 0.8 + 0.2
                    geom_new['rgba'] = [1, (1-a), (1-a), 1.0]
                geoms_new.append(geom_new)
            all_link_geoms[name] = geoms_new
            all_link_names.append(name)
    d['geoms'] = all_link_geoms
    d['link_names'] = all_link_names

    # stack states for the viewer
    statess_list = []
    for states in statess:
        states = jax.tree.map(lambda *x: jnp.concat(x), *states)
        statess_list.append(states)
    statess = jax.tree.map(lambda *x: jnp.stack(x), *statess_list)
    statess = _to_dict(statess)
    d['states'] = {k: statess[k] for k in ['x']}

    return json.dumps(d)

def render_us(state, us):
    rollout = []
    rew_sum = 0.0
    for i in range(Hsample):
        rollout.append(state.pipeline_state)
        state = step_env_jit(state, us[i])
        rew_sum += state.reward
    return rollout
    # rew_mean = rew_sum / (Hsample)
    # webpage = html.render(env.sys.replace(dt=env.dt), rollout)
    # print(f"evaluated reward mean: {rew_mean:.2e}")
    # with open(f"{path}/render_diffusion.html", "w") as f:
    #     f.write(webpage)

rng = jax.random.PRNGKey(0)
rng, rng_reset = jax.random.split(rng)
state_init = env.reset(rng_reset)
rollouts = []
for i in range(mu_0ts.shape[0]):
    rollout = render_us(state_init, mu_0ts[i])
    # rollouts.append([*rollout[::5], rollout[-1]])
    # rollouts.append([*rollout[::5]])
    # rollouts.append([*rollout[:40][::3]])
    rollouts.append([*rollout[:40][::3]])
json_file = dumps(env.sys.replace(dt=env.dt), rollouts)
html_file = render_from_json(json_file, height=500, colab=False, base_url=None)
with open(f"{path}/render_diffusion.html", "w") as f:
    f.write(html_file)