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

config.update("jax_enable_x64", True) # NOTE: this is important for simulating long horizon open loop control

## global config

use_data = True
init_data = False

## setup env

env_name = "halfcheetah"
backend = "positional"
if env_name in ["hopper", "walker2d"]:
    substeps = 10
elif env_name in ["humanoid", "humanoidstandup"]:
    substeps = 2
else:
    substeps = 1
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

Nexp = 8
Nsample = 1024
Hsample = 50
Ndiffuse = 100
# temp_sample = 0.5
temp_sample = 0.5
betas = jnp.linspace(1e-4, 1e-2, Ndiffuse)
alphas = 1.0 - betas
alphas_bar = jnp.cumprod(alphas)
sigmas = jnp.sqrt(1 - alphas_bar)
Sigmas_cond = (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
sigmas_cond = jnp.sqrt(Sigmas_cond)
sigmas_cond = sigmas_cond.at[0].set(0.0)
print(f"init sigma = {sigmas[-1]:.2e}")

Y0_hat_exp = jnp.zeros([Nexp, Hsample, Nu])
rng, rng_y = jax.random.split(rng)
Yt_exp = jax.random.normal(rng_y, (Nexp, Hsample, Nu))

if use_data:
    Y0_data = jnp.load(f"{path}/Y0.npy")
if init_data:
    Y0_data = jnp.load(f"{path}/Y0.npy")
    Y0_hat_exp = jnp.repeat(Y0_data[None], Nexp, axis=0)


# evaluate the diffused uss
@jax.jit
def eval_us(state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, state.reward

    _, rews = jax.lax.scan(step, state, us)
    return rews


def render_us(state, us):
    rollout = []
    rew_sum = 0.0
    for i in range(Hsample):
        for j in range(substeps):
            rollout.append(state.pipeline_state)
            state = step_env_jit(state, us[i])
            rew_sum += state.reward
    rew_mean = rew_sum / (Hsample * substeps)
    webpage = html.render(env.sys.replace(dt=env.dt), rollout)
    print(f"evaluated reward mean: {rew_mean:.2e}")
    with open(f"{path}/rollout.html", "w") as f:
        f.write(webpage)


@jax.jit
def reverse_once(carry, unused):
    t, rng, Y0_hat, Yt = carry

    # calculate Y0_hat
    # Method1: sampling around Y0_hat q(Y0)
    # sample Y0s from Y0_hat
    rng, Y0s_rng = jax.random.split(rng)
    eps_u = jax.random.normal(Y0s_rng, (Nsample, Hsample, Nu))
    Y0s = Y0_hat + eps_u * sigmas[t]

    if use_data:
        p_data = (t) / (Ndiffuse-1)
        rng, data_rng = jax.random.split(rng)
        data_mask = jax.random.bernoulli(data_rng, p=p_data, shape=(Nsample, Hsample, Nu))
        deltaY0 = Y0_data - Y0_hat
        Y0s = jnp.where(data_mask, Y0s + deltaY0, Y0s)
        Y0s = Y0s.at[0].set(Y0_data)
        eps_u = (Y0s - Y0_hat) / sigmas[t]

    Y0s = jnp.clip(Y0s, -1.0, 1.0)
    # calculate reward for Y0s
    eps_Y = (Y0s * jnp.sqrt(alphas_bar[t]) - Yt) / sigmas[t]
    logpdss = -0.5 * jnp.mean(eps_Y**2, axis=-1) + 0.5 * jnp.mean(eps_u**2, axis=-1)
    logpds = logpdss.mean(axis=-1)
    # logpds_normed = jnp.clip(logpds - logpds.max(), -1.0, 0.0)
    # logpds_normed = logpds
    rews = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0s).mean(axis=-1)
    rews_normed = (rews - rews.mean()) / rews.std()
    logweight = rews_normed + logpds
    weights = jax.nn.softmax(logweight / temp_sample)
    # jax.debug.print("max weight = {x} max rew={y} rew = {z} \pm {w}", x=weights.max(), y=rews.max(), z=rews.mean(), w=rews.std())
    # Get new Y0_hat
    Y0_hat_new = jnp.einsum("n,nij->ij", weights, Y0s)

    # Method2: sample around Yt P(Yt)
    # rng, Y0s_rng = jax.random.split(rng)
    # eps_Y = jax.random.normal(Y0s_rng, (Nsample, Hsample, Nu))
    # Y0s = Yt / jnp.sqrt(alphas_bar[t]) + eps_Y * jnp.sqrt(1 / alphas_bar[t] - 1)
    # Y0s = jnp.clip(Y0s, -1.0, 1.0)
    # # calculate reward for Y0s
    # rews = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0s).mean(axis=-1)
    # rews_normed = (rews - rews.mean()) / rews.std()
    # logweight = rews_normed
    # weights = jax.nn.softmax(logweight / temp_sample)
    # # Get new Y0_hat
    # Y0_hat_new = jnp.einsum("n,nij->ij", weights, Y0s)

    # calculate score function
    ky = -1.0 / (1 - alphas_bar[t])
    kx = jnp.sqrt(alphas_bar[t]) / (1 - alphas_bar[t])
    score = ky * Yt + kx * Y0_hat_new

    # calculate Ytm1
    rng, Ytm1_rng = jax.random.split(rng)
    eps_Ytm1 = jax.random.normal(Ytm1_rng, (Hsample, Nu))
    Ytm1 = (
        1 / jnp.sqrt(1.0 - betas[t]) * (Yt + 0.5 * betas[t] * score)
        + jnp.sqrt(betas[t]) * eps_Ytm1
    )

    return (t - 1, rng, Y0_hat_new, Ytm1), rews.mean()


# run reverse
def reverse(Y0_hat, Yt, rng):
    carry_once = (Ndiffuse - 1, rng, Y0_hat, Yt)
    (t0, rng, Y0_hat, Y0), rew = jax.lax.scan(reverse_once, carry_once, None, Ndiffuse)
    return Y0_hat, Y0, rew


rng_exp = jax.random.split(rng, Nexp)
Y0_hat_exp, Y0_exp, rew_exp = jax.vmap(reverse)(Y0_hat_exp, Yt_exp, rng_exp)

rew_eval = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0_exp).mean(axis=-1)
print(f"rews mean: {rew_eval.mean():.2e} std: {rew_eval.std():.2e}")

render_us(state_init, Y0_exp[jnp.argmax(rew_eval)])