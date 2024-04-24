import functools
from datetime import datetime
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model, html
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt

## setup env

env_name = "ant"
backend = "spring"
env = envs.get_environment(env_name=env_name, backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

## load model
jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)

rng = jax.random.PRNGKey(seed=0)
reset_rng, rng = jax.random.split(rng)
state = jit_env_reset(rng=reset_rng)
"""

load_backend = "positional"
params = model.load_params(f"../figure/{env_name}/{load_backend}/params")
normalize = running_statistics.normalize
ppo_network = ppo_networks.make_ppo_networks(
    state.obs.shape[-1], env.action_size, preprocess_observations_fn=normalize
)
make_policy = ppo_networks.make_inference_fn(ppo_network)
inference_fn = make_policy(params)
jit_inference_fn = jax.jit(inference_fn)

reward_sum = 0.0
Hrollout = 50
us_policy = jnp.zeros([Hrollout, env.action_size])
rollout = []
for i in range(Hrollout):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    us_policy = us_policy.at[i].set(act)
    state = jit_env_step(state, act)
    reward_sum += state.reward
print(f"reward_sum: {reward_sum}")
# webpage = html.render(env.sys.replace(dt=env.dt), rollout)
# # save it to a file
# with open(f"../figure/{env_name}/{load_backend}/{backend}_render.html", "w") as f:
#     f.write(webpage)
"""

## run MPPI
Nmppi = 1024 * 16
Nnode = 50
Hnode = 1
Hmppi = (Nnode - 1) * Hnode + 1
nx = env.observation_size
nu = env.action_size
temp_mppi = 0.1
sigmas = 0.2 / (jnp.arange(20) + 1)
# us_node = us_policy[::Hnode]
us_node = jnp.zeros([Nnode, nu])
# us_node = us_policy[::Hnode]
y_rng, rng = jax.random.split(rng)
yus_node = us_node + jax.random.normal(y_rng, us_node.shape) * sigmas[0]


@jax.jit
def eval_us(state, us):
    def step(state, u):
        state = jit_env_step(state, u)
        return state, state.reward

    _, rews = jax.lax.scan(step, state, us)
    return rews


# rollout policy for initial u sequence
mppi_rng, rng = jax.random.split(rng)


def linear_interpolation_single(x0, x1):
    return jnp.linspace(x0, x1, Hnode + 1)[:-1]


@jax.jit
def linear_interpolation(us_node):
    us_node0 = us_node[:-1]
    us_node1 = us_node[1:]
    us_wolast = jax.vmap(linear_interpolation_single, in_axes=(0, 0))(
        us_node0, us_node1
    )
    us_wolast = jnp.reshape(us_wolast, (Hmppi - 1, nu))
    us = jnp.concatenate([us_wolast, us_node[-1:]], axis=0)
    return us


fig, axes = plt.subplots(1, 2)
state = jit_env_reset(rng=reset_rng)
reward_sum = 0.0
for i in range(0, sigmas.shape[0]):
    sigma = sigmas[i]
    # sigma_prev = sigmas[i - 1]
    # alpha_bar = 1 - sigma**2
    # alpha_bar_prev = 1 - sigma_prev**2
    # alpha = alpha_bar_prev / alpha_bar
    # var_cond = (1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)

    rng, mppi_rng = jax.random.split(rng)
    uss_node = jax.random.normal(mppi_rng, (Nmppi, Nnode, nu)) * sigma + us_node
    uss_node = jnp.clip(uss_node, -1.0, 1.0)
    uss = jax.vmap(linear_interpolation, in_axes=(0))(uss_node)
    uss = jnp.clip(uss, -1.0, 1.0)

    rewss = jax.vmap(eval_us, in_axes=(None, 0))(state, uss)
    rews = jnp.mean(rewss, axis=-1)
    rews_normed = (rews - rews.mean()) / rews.std()

    weights = jax.nn.softmax(rews_normed / temp_mppi)
    # us = jnp.einsum("n,nij->ij", weights, uss)
    # for j in range(Hnode):
    #     state = jit_env_step(state, us[j])
    #     rollout.append(state.pipeline_state)
    #     reward_sum += state.reward
    us_node = jnp.einsum("n,nij->ij", weights, uss_node)

    ys_rng, rng = jax.random.split(rng)
    yus_node = (
        jnp.sqrt(alpha) * (1 - alpha_bar_prev) * yus_node
        + jnp.sqrt(alpha_bar_prev) * (1 - alpha) * us_node
    ) / (1 - alpha_bar) + jax.random.normal(ys_key, (Nnode, nu)) * jnp.sqrt(var_cond)

    print(f"rew={rews.mean():.2f} pm {rews.std():.2f}")
    print(f"weight={weights.mean():.2f} pm {weights.std():.2f}")
    # plot histogram
    # axes[0].cla()
    # axes[0].hist(rews, bins=100)
    # axes[1].cla()
    # axes[1].hist(weights, bins=20)
    # axes[1].set_ylim([0, 100])
    # plt.pause(0.01)

us = linear_interpolation(us_node)
rollout = []
state = jit_env_reset(rng=reset_rng)
for i in range(Hmppi):
    rollout.append(state.pipeline_state)
    state = jit_env_step(state, us[i])
    reward_sum += state.reward

print(f"reward_sum: {reward_sum}")
webpage = html.render(env.sys.replace(dt=env.dt), rollout)
with open(f"../figure/{env_name}/{backend}/mppi_render.html", "w") as f:
    f.write(webpage)
