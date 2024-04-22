import functools
import os
from datetime import datetime
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model, html
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
from jax import config

config.update("jax_enable_x64", True)

## setup env

env_name = "hopper"
backend = "positional"
env = envs.get_environment(env_name=env_name, backend=backend)
Nx = env.observation_size
Nu = env.action_size
step_env = jax.jit(env.step)
reset_env = jax.jit(env.reset)
rng = jax.random.PRNGKey(seed=0)
rng, rng_reset = jax.random.split(rng)  # NOTE: rng_reset should never be changed.
state_init = reset_env(rng_reset)

## run diffusion

Nexp = 1
Nsample = 1024
Hsample = 50
Ndiffuse = 100
temp_sample = 0.5  # 0.1
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
        rollout.append(state.pipeline_state)
        state = step_env(state, us[i])
        rew_sum += state.reward
    webpage = html.render(env.sys.replace(dt=env.dt), rollout)
    print(f"evaluated reward mean: {(rew_sum / Hsample):.2e}")
    path = f"../figure/{env_name}/{backend}"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/rollout.html", "w") as f:
        f.write(webpage)

@jax.jit
def reverse_once(carry, unused):
    t, rng, Y0_hat, Yt = carry

    # calculate Y0_hat
    # sample Y0s from Y0_hat
    rng, Y0s_rng = jax.random.split(rng)
    eps_u = jax.random.normal(Y0s_rng, (Nsample, Hsample, Nu))
    Y0s = Y0_hat + eps_u * sigmas[t]
    Y0s = jnp.clip(Y0s, -1.0, 1.0)
    # calculate reward for Y0s
    rews = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0s).mean(axis=-1)
    logweight = (rews - rews.mean()) / rews.std()
    weights = jax.nn.softmax(logweight / temp_sample)
    # Get new Y0_hat
    Y0_hat_new = jnp.einsum("n,nij->ij", weights, Y0s)

    # calculate score function
    ky = jnp.sqrt(alphas[t]) * (1 - alphas_bar[t-1]) / (1 - alphas_bar[t])
    kx = jnp.sqrt(alphas_bar[t-1]) * (1 - alphas[t]) / (1 - alphas_bar[t])
    score = (ky - 1) * Yt + kx * Y0_hat_new

    # calculate Ytm1
    rng, Ytm1_rng = jax.random.split(rng)
    eps_Ytm1 = jax.random.normal(Ytm1_rng, (Hsample, Nu))
    Ytm1 = Yt + score + sigmas_cond[t] * eps_Ytm1

    return (t - 1, rng, Y0_hat_new, Ytm1), rews.mean()

# run reverse
def reverse(Y0_hat, Yt, rng):
    carry_once = (Ndiffuse - 1, rng, Y0_hat, Yt)
    (t0, rng, Y0_hat, Y0), rew = jax.lax.scan(reverse_once, carry_once, None, Ndiffuse)
    return Y0_hat, Y0, rew

rng_exp = jax.random.split(rng, Nexp)
Y0_hat_exp, Y0_exp, rew_exp = jax.vmap(reverse)(Y0_hat_exp, Yt_exp, rng_exp)

print(f"rews mean: {rew_exp.mean():.2e} std: {rew_exp.std():.2e}")

render_us(state_init, Y0_hat_exp[jnp.argmax(rew_exp)])