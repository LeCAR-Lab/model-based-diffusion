import os
import functools
from datetime import datetime
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import train as sac
from brax.training.agents.sac import networks as sac_networks
from brax.training.acme import running_statistics
from brax.io import model, html
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
from jax import config

# config.update("jax_enable_x64", True) # NOTE: this is important for simulating long horizon open loop control


## setup env

env_name = "pushT"
backend = "positional"
if env_name == "pushT":
    from pushT import PushT
    env = PushT()
else:
    env = envs.get_environment(env_name=env_name, backend=backend)
rng = jax.random.PRNGKey(seed=0)
rng, rng_reset = jax.random.split(rng)
state = jax.jit(env.reset)(rng=rng_reset)

## train
if env_name in ['ant', 'pusher', 'halfcheetah', 'pusher', 'humanoid', 'humanoidstandup', 'pushT']:
    normalize = running_statistics.normalize
    ppo_network = ppo_networks.make_ppo_networks(
        state.obs.shape[-1], env.action_size, preprocess_observations_fn=normalize
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
else:
    normalize = running_statistics.normalize
    sac_network = sac_networks.make_sac_networks(
        state.obs.shape[-1], env.action_size, preprocess_observations_fn=normalize
    )
    make_inference_fn = sac_networks.make_inference_fn(sac_network)

path = f"../figure/{env_name}/{backend}"
params = model.load_params(f"{path}/params")


## evaluate

inference_fn = make_inference_fn(params)

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
state = jit_env_reset(rng=rng_reset)
reward_sum = 0
if env_name in ['hopper', 'walker2d']:
    Heval = 500
    substeps = 10
elif env_name in ['humanoid', 'humanoidstandup']:
    Heval = 100
    substeps = 2
elif env_name in ["pushT"]:
    Heval = 50
    substeps = 1
else:
    Heval = 50
    substeps = 1
us_policy = []
for _ in range(Heval):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)
    reward_sum += state.reward
    us_policy.append(act)
print(f"evaluated reward mean: {(reward_sum / Heval):.2e}")
us_policy = jnp.stack(us_policy)
# jnp.save(f"{path}/us_policy.npy", us_policy)
webpage = html.render(env.sys.replace(dt=env.dt), rollout)
with open(f"{path}/RL.html", "w") as f:
    f.write(webpage)

## run us_policy in new backend
backend_test = "positional"
noise = 0.0
if env_name == "pushT":
    env_test = PushT()
else:
    env_test = envs.get_environment(env_name=env_name, backend=backend_test)
state_test = jax.jit(env_test.reset)(rng=rng_reset)
jit_env_step_test = jax.jit(env_test.step)
rollout = []
Y0 = []
reward_sum = 0
for i in range(Heval):
    if i % substeps == 0:
        u = us_policy[i]
        # if substeps == 1:
        rng, rng_noise = jax.random.split(rng)
        u = u + noise * jax.random.normal(rng_noise, u.shape)
        Y0.append(u)
    rollout.append(state_test.pipeline_state)
    state_test = jit_env_step_test(state_test,u)
    reward_sum += state_test.reward
Y0 = jnp.stack(Y0)
jnp.save(f"{path}/Y0.npy", Y0)
print(f"evaluated reward mean (new backend, openloop): {(reward_sum / Heval):.2e}")
webpage = html.render(env_test.sys.replace(dt=env_test.dt), rollout)
with open(f"{path}/openloop.html", "w") as f:
    f.write(webpage)