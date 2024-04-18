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

env_name = 'ant'
backend = 'positional'
env = envs.get_environment(env_name=env_name,
                           backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

"""
## train
train_fn = {
  'ant': functools.partial(ppo.train,  num_timesteps=50_000_000, num_evals=10, reward_scaling=10, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=4096, batch_size=2048, seed=1),
}[env_name]

max_y = {'ant': 8000}[env_name]
min_y = {'ant': 0}[env_name]

fig, ax = plt.subplots()
xdata, ydata = [], []
times = [datetime.now()]

def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    ax.clear()
    ax.set_xlim([0, train_fn.keywords['num_timesteps']])
    ax.set_ylim([min_y, max_y])
    ax.set_xlabel('# environment steps')
    ax.set_ylabel('reward per episode')
    ax.plot(xdata, ydata)
    plt.pause(0.01)

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

model.save_params(f'../figure/{env_name}/{backend}/params', params)
"""

## load model

load_backend = 'positional'
params = model.load_params(f'../figure/{env_name}/{load_backend}/params')
jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)

rng = jax.random.PRNGKey(seed=1)
state_reset = jit_env_reset(rng=rng)

normalize = running_statistics.normalize
ppo_network = ppo_networks.make_ppo_networks(
    state.obs.shape[-1],
    env.action_size,
    preprocess_observations_fn=normalize)
make_policy = ppo_networks.make_inference_fn(ppo_network)
inference_fn = make_policy(params)
jit_inference_fn = jax.jit(inference_fn)

"""
reward_sum = 0.0
rollout = []
for _ in range(1000):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)
    reward_sum += state.reward
print(f"reward_sum: {reward_sum}")
webpage = html.render(env.sys.replace(dt = env.dt), rollout)
# save it to a file
with open(f'../figure/{env_name}/{load_backend}/{backend}_render.html', 'w') as f:
    f.write(webpage)

"""
## run MPPI
Hmppi = 100
Nmppi = 100

def eval_us(state, us):
    def step(state, u):
        state = jit_env_step(state, u)
        return state, state.reward

    _, rews = jax.lax.scan(step, state, us)
    return rews.mean()

# rollout policy for initial u sequence
state = state_reset
act_rng, rng = jax.random.split(rng)
us_policy = jnp.zeros([Hmppi, env.action_size])
reward_policy = 0.0
for i in range(Hmppi):
    act, _ = jit_inference_fn(state.obs, act_rng)
    us_policy = us_policy.at[i, :].set(act)
    state = jit_env_step(state, act)
    reward_policy += state.reward
# plot us_policy with matplotlib
plt.plot(us_policy[:, 0], label = "policy")
plt.legend()
plt.show()
exit()

reward_policy = reward_policy / Hmppi
mppi_rng, rng = jax.random.split(rng)
uss = jax.random.normal(mppi_rng, (Nmppi, Hmppi, env.action_size)) * 0.2 + us_policy[None, ...]
rewards_mppi = jax.vmap(functools.partial(eval_us, state_reset))(uss)
print(f"policy reward mean = {reward_policy}")
print(f"mppi reward mean = {rewards_mppi.mean()}, std = {rewards_mppi.std()}, max = {rewards_mppi.max()}")
