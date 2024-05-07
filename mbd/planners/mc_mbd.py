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
from dataclasses import dataclass
import tyro

import mbd

# NOTE: this is important for simulating long horizon open loop control
# config.update("jax_enable_x64", True)


## load config
@dataclass
class Args:
    # exp
    seed: int = 0
    silent: bool = False
    # env
    env_name: str = "ant"
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 50  # horizon
    Ndiffuse: int = 100  # number of diffusion steps
    temp_sample: float = 0.1  # temperature for sampling
    beta0: float = 1e-4  # initial beta
    betaT: float = 1e-2  # final beta


args = tyro.cli(Args)
rng = jax.random.PRNGKey(seed=args.seed)

## setup env

# env_name = "humanoidstandup"
# recommended temperature for envs
temp_dict = {
    "ant": 0.1,
    "halfcheetah": 0.4, 
}
env = mbd.get_env(args.env_name)
Nx = env.observation_size
Nu = env.action_size
step_env_jit = jax.jit(env.step)
reset_env_jit = jax.jit(env.reset)

rng, rng_reset = jax.random.split(rng)  # NOTE: rng_reset should never be changed.
state_init = reset_env_jit(rng_reset)

path = f"{mbd.__path__[0]}/../results/{args.env_name}"
if not os.path.exists(path):
    os.makedirs(path)

## run diffusion

betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
alphas = 1.0 - betas
alphas_bar = jnp.cumprod(alphas)
sigmas = jnp.sqrt(1 - alphas_bar)
Sigmas_cond = (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
sigmas_cond = jnp.sqrt(Sigmas_cond)
sigmas_cond = sigmas_cond.at[0].set(0.0)
if not args.silent:
    print(f"init sigma = {sigmas[-1]:.2e}")

mu_0T = jnp.zeros([args.Hsample, Nu])


# evaluate the diffused uss
@jax.jit
def eval_us(state, us):
    def step(state, u):
        state = step_env_jit(state, u)
        return state, state.reward

    _, rews = jax.lax.scan(step, state, us)
    return rews


def render_us(state, us):
    rollout = []
    rew_sum = 0.0
    for i in range(args.Hsample):
        rollout.append(state.pipeline_state)
        state = step_env_jit(state, us[i])
        rew_sum += state.reward
    rew_mean = rew_sum / (args.Hsample)
    webpage = html.render(env.sys.replace(dt=env.dt), rollout)
    print(f"evaluated reward mean: {rew_mean:.2e}")
    with open(f"{path}/rollout.html", "w") as f:
        f.write(webpage)


@jax.jit
def reverse_once(carry, unused):
    t, rng, mu_0t = carry

    # sample from q_i
    rng, Y0s_rng = jax.random.split(rng)
    eps_u = jax.random.normal(Y0s_rng, (args.Nsample, args.Hsample, Nu))
    Y0s = eps_u * sigmas[t] + mu_0t
    Y0s = jnp.clip(Y0s, -1.0, 1.0)

    # esitimate mu_0tm1
    rews = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0s).mean(axis=-1)
    logp0 = (rews - rews.mean()) / rews.std() / args.temp_sample
    weights = jax.nn.softmax(logp0)
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)  # NOTE: update only with reward

    return (t - 1, rng, mu_0tm1), rews.mean()


# run reverse
def reverse(mu_0T, rng):
    mu_0t = mu_0T
    mu_0ts = []
    for t in range(args.Ndiffuse - 1, 0, -1):
        carry_once = (t, rng, mu_0t)
        (t, rng, mu_0t), rew = reverse_once(carry_once, None)
        mu_0ts.append(mu_0t)
        if not args.silent:
            print(f"t: {t}, rew: {rew:.2e}")
    return mu_0ts


rng_exp, rng = jax.random.split(rng)
mu_0ts = reverse(mu_0T, rng_exp)
mu_0ts = jnp.array(mu_0ts)
jnp.save(f"{path}/mu_0ts.npy", mu_0ts)
render_us(state_init, mu_0ts[-1])
