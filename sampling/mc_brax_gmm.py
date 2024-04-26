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

env_name = "pushT"
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

Nexp = 16
Nsample = 1024 * 4
Hsample = 50
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

rng, rng_y0 = jax.random.split(rng)
Y0s_hat = jax.random.normal(rng_y0, (Nsample, Hsample, Nu))
weights_Y0_hat = jnp.ones(Nsample) / Nsample
rng, rng_y = jax.random.split(rng)
Yts = jax.random.normal(rng_y, (Nexp, Hsample, Nu))


# evaluate the diffused uss
@jax.jit
def eval_us(state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, state.reward

    _, rews = jax.lax.scan(step, state, us)
    return rews


def render_us(state, us, name="rollout"):
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
    with open(f"{path}/{name}.html", "w") as f:
        f.write(webpage)


@jax.jit
def reverse_once(carry, unused):
    t, rng, Y0s_hat, weights_Y0_hat, Yts = carry

    # Method1: sampling around Y0_hat q(Y0)
    # sample Y0s from Y0s_hat
    rng, Y0s_rng = jax.random.split(rng)
    eps_u = jax.random.normal(Y0s_rng, (Nsample, Hsample, Nu))
    Y0s = eps_u * sigmas[t] + Y0s_hat  # (Nsample, Hsample, Nu)
    logpdss_Y0 = -0.5 * jnp.mean(eps_u**2, axis=-1)

    Y0s = jnp.clip(Y0s, -1.0, 1.0)
    # calculate reward for Y0s
    eps_Y = jnp.clip(
        (Y0s[None] * jnp.sqrt(alphas_bar[t]) - Yts[:, None]) / sigmas[t], -2.0, 2.0
    )  # (Nexp, Nsample, Hsample, Nu)
    logpdss_Yt = -0.5 * jnp.mean(eps_Y**2, axis=-1)  # (Nexp, Nsample, Hsample)
    logpdss = logpdss_Yt - logpdss_Y0
    logpds = logpdss.mean(axis=-1)  # (Nexp, Nsample)
    rews = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0s).mean(axis=-1)

    jax.debug.print("rews={x} \pm {y}", x=rews.mean(), y=rews.std())

    rews_normed = (rews - rews.mean()) / rews.std()
    logweight = rews_normed + logpds
    weights_Y0_bar = jax.nn.softmax(logweight / temp_sample, axis=-1)  # (Nexp, Nsample)
    weights_Y0_hat_new = jax.nn.softmax(
        rews_normed / temp_sample + jnp.log(weights_Y0_hat), axis=-1
    )  # (Nexp, Nsample)
    std_w_rew = jnp.std(weights_Y0_hat_new, axis=-1).mean()
    jax.debug.print("std_w_rew={x}", x=std_w_rew)
    Y0s_bar = jnp.einsum("mn,nij->mij", weights_Y0_bar, Y0s)  # (Nexp, Hsample, Nu)
    rng, idx_rng = jax.random.split(rng)
    need_resample = std_w_rew > 1e-4
    idx = jax.random.categorical(idx_rng, rews_normed / temp_sample, shape=(Nsample,))
    Y0s_hat_new_resample = Y0s[idx]  # (Nsample, Hsample, Nu)
    weights_Y0_hat_new_resample = jnp.ones(Nsample) / Nsample
    Y0s_hat_new = need_resample * Y0s_hat_new_resample + (1 - need_resample) * Y0s
    weights_Y0_hat_new = (
        need_resample * weights_Y0_hat_new_resample
        + (1 - need_resample) * weights_Y0_hat_new
    )

    # calculate score function
    scores = (
        alphas_bar[t] / (1 - alphas_bar[t]) * (Y0s_bar - Yts / jnp.sqrt(alphas_bar[t]))
    )

    # calculate Ytm1
    rng, Ytm1_rng = jax.random.split(rng)
    eps_Ytm1 = jax.random.normal(Ytm1_rng, (Nexp, Hsample, Nu))
    Ytsm1 = (
        1 / jnp.sqrt(1.0 - betas[t]) * (Yts + 0.5 * betas[t] * scores)
        + 1.0 * jnp.sqrt(betas[t]) * eps_Ytm1
    )

    return (t - 1, rng, Y0s_hat_new, weights_Y0_hat_new, Ytsm1), rews.mean()


# run reverse
def reverse(Y0s_hat, weights_Y0_hat, Yts, rng):
    carry_once = (Ndiffuse - 1, rng, Y0s_hat, weights_Y0_hat, Yts)
    (t0, rng, Y0s_hat, weights_Y0_hat, Y0s), rew = jax.lax.scan(
        reverse_once, carry_once, None, Ndiffuse
    )
    # for i in range(Ndiffuse):
    #     carry_once, rew = reverse_once(carry_once, None)
    # (tT, rng, Y0_hat, Y0), rew = carry_once
    rng, rng_Y0 = jax.random.split(rng)
    idx = jax.random.categorical(rng_Y0, weights_Y0_hat, shape=(Nsample,))
    Y0s_hat = Y0s_hat[idx]
    return Y0s_hat, Y0s, rew


rng, rng_exp = jax.random.split(rng)
Y0s_hat, Y0s, rew_exp = reverse(Y0s_hat, weights_Y0_hat, Yts, rng_exp)
rew_eval = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0s_hat).mean(axis=-1)
print(f"rews mean: {rew_eval.mean():.2e} std: {rew_eval.std():.2e}")

for j in range(8):
    render_us(state_init, Y0s_hat[j], f"rollout{j}")
