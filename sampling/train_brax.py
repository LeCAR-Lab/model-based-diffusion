import os
import functools
from datetime import datetime
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from brax.io import model, html
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
from jax import config
import argparse

## setup env
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="halfcheetah")
parser.add_argument("--backend", type=str, default="positional")
args = parser.parse_args()
env_name = args.env_name
backend = args.backend
if env_name == "pushT":
    from pushT import PushT
    env = PushT()
else:
    env = envs.get_environment(env_name=env_name, backend=backend)
rng = jax.random.PRNGKey(seed=0)
rng, rng_reset = jax.random.split(rng)
state = jax.jit(env.reset)(rng=rng_reset)

## train
train_fn = {
    "ant": functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=10,
        reward_scaling=10,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=4096,
        batch_size=2048,
        seed=1,
    ),
    "hopper": functools.partial(
        sac.train,
        num_timesteps=6_553_600,
        num_evals=20,
        reward_scaling=30,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        discounting=0.997,
        learning_rate=6e-4,
        num_envs=128,
        batch_size=512,
        grad_updates_per_step=64,
        max_devices_per_host=1,
        max_replay_size=1048576,
        min_replay_size=8192,
        seed=1,
    ),
    "walker2d": functools.partial(
        sac.train,
        num_timesteps=7_864_320,
        num_evals=20,
        reward_scaling=5,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        discounting=0.997,
        learning_rate=6e-4,
        num_envs=128,
        batch_size=128,
        grad_updates_per_step=32,
        max_devices_per_host=1,
        max_replay_size=1048576,
        min_replay_size=8192,
        seed=1,
    ),
    "halfcheetah": functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=20,
        reward_scaling=1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=0.001,
        num_envs=2048,
        batch_size=512,
        seed=3,
    ),
    "pusher": functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=20,
        reward_scaling=5,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=30,
        num_minibatches=16,
        num_updates_per_batch=8,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=2048,
        batch_size=512,
        seed=3,
    ),
    "pushT": functools.partial(
        ppo.train,
        num_timesteps=10_000_000,
        num_evals=20,
        reward_scaling=5,
        episode_length=100,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=16,
        num_updates_per_batch=8,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=2048,
        batch_size=512,
        seed=0,
    ),
    "humanoid": functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=10,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=1024,
        seed=1,
    ),
    "humanoidstandup": functools.partial(
        ppo.train,
        num_timesteps=100_000_000,
        num_evals=20,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=15,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=6e-4,
        entropy_cost=1e-2,
        num_envs=2048,
        batch_size=1024,
        seed=1,
    ),
}[env_name]

max_y = {
    "ant": 8000,
    "halfcheetah": 8000,
    "hopper": 2500,
    "humanoid": 13000,
    "humanoidstandup": 75_000,
    "reacher": 5,
    "walker2d": 5000,
    "pusher": 0,
    "pushT": 100, 
}[env_name]
min_y = {"reacher": -100, "pusher": -150, "pushT": -50}.get(env_name, 0)

fig, ax = plt.subplots()
xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics["eval/episode_reward"])
    ax.clear()
    ax.set_xlim([0, train_fn.keywords["num_timesteps"]])
    ax.set_ylim([min_y, max_y])
    ax.set_xlabel("# environment steps")
    ax.set_ylabel("reward per episode")
    ax.plot(xdata, ydata)
    plt.pause(0.01)


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

path = f"../figure/{env_name}/{backend}"
if not os.path.exists(path):
    os.makedirs(path)
model.save_params(f"{path}/params", params)


## evaluate

# @title Visualizing a trajectory of the learned inference function

# create an env with auto-reset
# env = envs.create(env_name=env_name, backend=backend)

inference_fn = make_inference_fn(params)

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
state = jit_env_reset(rng=rng_reset)
for _ in range(1000):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)

webpage = html.render(env.sys.replace(dt=env.dt), rollout)
with open(f"{path}/RL.html", "w") as f:
    f.write(webpage)