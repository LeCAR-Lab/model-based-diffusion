import functools
import os
from datetime import datetime
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model, html
import jax
from jax import lax
from jax import numpy as jnp
from matplotlib import pyplot as plt
from jax import config

# config.update("jax_enable_x64", True) # NOTE: this is important for simulating long horizon open loop control

## global config

use_data = False
init_data = False

## setup env

env_name = "car"
backend = "positional"
if env_name in ["hopper", "walker2d"]:
    substeps = 10
elif env_name in ["humanoid", "humanoidstandup"]:
    substeps = 2
else:
    substeps = 1
if env_name == "pushT":
    from pushT import PushT

    env = PushT()
elif env_name == "point":
    from point import Point, State, vis_env

    env = Point()

    fig, ax = plt.subplots()

    # load demostration from ../figure/rrt_path.npy
    xs_demo = jnp.zeros([31, 2]) + env.x_goal
    xs_demo_load = jnp.load("../figure/rrt_path.npy")
    xs_demo = xs_demo.at[: xs_demo_load.shape[0]].set(xs_demo_load)
elif env_name == "car":
    from car import Car, State, vis_env

    env = Car()

    fig, ax = plt.subplots()

    # load demostration from ../figure/rrt_path.npy
    # x_mid = jnp.array([0.0, 2.0])
    # xs_demo_load = jnp.concatenate(
    #     [
    #         jnp.linspace(jnp.array([-1.0, 0.0]), x_mid, 10),
    #         jnp.linspace(x_mid, jnp.array([1.0, 0.0]), 10),
    #     ]
    # )
    xs_demo = jnp.zeros([31, 4]) + env.x_goal
    xs_demo_load = jnp.load("../figure/rrt_path.npy")
    xs_demo = xs_demo.at[: xs_demo_load.shape[0], : xs_demo_load.shape[1]].set(
        xs_demo_load
    )
else:
    env = envs.get_environment(env_name=env_name, backend=backend)
Nx = env.observation_size
Nu = env.action_size
step_env_jit = jax.jit(env.step)

if substeps > 1:

    @jax.jit
    def step_env(state, u):
        def step_once(state, unused):
            return step_env_jit(state, u), state.reward

        state, rews = lax.scan(step_once, state, None, length=substeps)
        state = state.replace(reward=rews.mean())
        return state

else:
    step_env = step_env_jit

reset_env = jax.jit(env.reset)
rng = jax.random.PRNGKey(seed=0)
rng, rng_reset = jax.random.split(rng)  # NOTE: rng_reset should never be changed.
state_init = reset_env(rng_reset)
path = f"../figure/{env_name}/{backend}"
if not os.path.exists(path):
    os.makedirs(path)

## run diffusion

Nexp = 1
Nsample = 1024
Hsample = 50
if env_name == "point" or "car":
    Nsample = 128
    Hsample = 30
Ndiffuse = 100
temp_sample = 0.5
beta0 = 1e-4
betaT = 1e-2
betas = jnp.linspace(beta0, betaT, Ndiffuse)
alphas = 1.0 - betas
alphas_bar = jnp.cumprod(alphas)
sigmas = jnp.sqrt(1 - alphas_bar)
Sigmas_cond = (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
sigmas_cond = jnp.sqrt(Sigmas_cond)
sigmas_cond = sigmas_cond.at[0].set(0.0)
print(f"init sigma = {sigmas[-1]:.2e}")

mu_0T = jnp.zeros([Nexp, Hsample, Nu])


# evaluate the diffused uss
@jax.jit
def eval_us(state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, state.reward

    _, rews = jax.lax.scan(step, state, us)
    return rews


@jax.jit
def rollout_us(state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, state.obs

    _, rollout = jax.lax.scan(step, state, us)
    return rollout


# @jax.jit
def eval_xs_full(xs, us):
    states = jax.vmap(lambda x: State(obs=x, reward=0.0, done=0.0))(xs[:-1])
    states_next = jax.vmap(step_env, in_axes=(0, 0))(states, us)
    xs_next = states_next.obs
    return -((xs[1:] - xs_next) ** 2).sum()


def eval_xs_partial(xs, us):
    state0 = State(obs=xs[0], reward=0.0, done=0.0)
    xs_rollout = rollout_us(state0, us)
    return -((xs[1:, :2] - xs_rollout[:, :2]) ** 2).sum()


eval_xs = eval_xs_partial


def render_us(state, us):
    rollout = []
    rew_sum = 0.0
    for i in range(Hsample):
        for j in range(substeps):
            rollout.append(state.obs)
            state = step_env_jit(state, us[i])
            rew_sum += state.reward
    # rew_mean = rew_sum / (Hsample * substeps)
    # xs = jnp.stack([jnp.stack(rollout), xs_demo])
    xs = jnp.stack(rollout)[None]
    vis_env(ax, xs)
    ax.plot(xs_demo[:, 0], xs_demo[:, 1], "r--")
    # print(f"evaluated reward mean: {rew_mean}")


@jax.jit
def reverse_once(carry, unused):
    t, rng, mu_0t = carry

    # sample from q_i
    rng, Y0s_rng = jax.random.split(rng)
    eps_u = jax.random.normal(Y0s_rng, (Nsample, Hsample, Nu))
    Y0s = eps_u * sigmas[t] + mu_0t
    Y0s = jnp.clip(Y0s, -1.0, 1.0)

    # esitimate mu_0tm1
    rews = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0s).mean(axis=-1)
    # rews_std = jnp.where(jnp.isnan(rews.std()), 1.0, rews.std())
    # logpJ = (rews - rews.mean()) / rews_std / temp_sample
    logpJ = rews / temp_sample
    value_xs = jax.vmap(eval_xs, in_axes=(None, 0))(xs_demo, Y0s)
    logpdemo = value_xs - jnp.mean(value_xs) + logpJ
    jax.debug.print("rews={x} \pm {y}", x=rews.mean(), y=rews.std())
    jax.debug.print("logpdemo={x} \pm {y}", x=logpdemo.mean(), y=logpdemo.std())
    # logp0 = jnp.concat([logpJ, logpJ], axis=0)
    logp0 = jnp.concat([logpJ, logpdemo], axis=0)
    weights = jax.nn.softmax(logp0)
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, jnp.concatenate([Y0s, Y0s], axis=0))

    return (t - 1, rng, mu_0tm1), rews.mean()


# run reverse
def reverse(mu_0T, rng):
    carry_once = (Ndiffuse - 1, rng, mu_0T)
    # (_, rng, mu_0T), rew = jax.lax.scan(reverse_once, carry_once, None, Ndiffuse)
    for i in range(Ndiffuse):
        carry_once, rew = reverse_once(carry_once, None)
        render_us(state_init, carry_once[2])
        plt.savefig(f"{path}/{i}.png")
        plt.pause(0.1)
    return mu_0T, rew


rng_exp = jax.random.split(rng, Nexp)
mu_0, rew_exp = reverse(mu_0T[0], rng_exp[0])
rew_eval = jax.vmap(eval_us, in_axes=(None, 0))(state_init, mu_0).mean(axis=-1)
print(f"rews mean: {rew_eval.mean():.2e} std: {rew_eval.std():.2e}")

render_us(state_init, mu_0[jnp.argmax(rew_eval)])
