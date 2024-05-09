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
from tqdm import tqdm

import mbd

# NOTE: this is important for simulating long horizon open loop control
# config.update("jax_enable_x64", True)


## load config
@dataclass
class Args:
    # exp
    seed: int = 0
    disable_recommended_params: bool = False
    render: bool = True
    # env
    env_name: str = "ant" # "humanoidstandup", "ant", "halfcheetah", "hopper", "walker2d"
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 50  # horizon
    Ndiffuse: int = 100  # number of diffusion steps
    temp_sample: float = 0.1  # temperature for sampling
    beta0: float = 1e-4  # initial beta
    betaT: float = 1e-2  # final beta


def run_diffusion(args: Args):

    rng = jax.random.PRNGKey(seed=args.seed)

    ## setup env

    # recommended temperature for envs
    temp_recommend = {
        "ant": 0.1,
        "halfcheetah": 0.4,
        "hopper": 0.1,
        "humanoidstandup": 0.1,
        "humanoidrun": 0.1,
        "walker2d": 0.1,
    }
    if not args.disable_recommended_params:
        args.temp_sample = temp_recommend.get(args.env_name, args.temp_sample)
        print(f"override temp_sample to {args.temp_sample}")
    env = mbd.envs.get_env(args.env_name)
    Nx = env.observation_size
    Nu = env.action_size
    # env functions
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    eval_us = jax.jit(functools.partial(mbd.utils.eval_us, step_env_jit))
    render_us = functools.partial(
        mbd.utils.render_us, step_env_jit, env.sys.replace(dt=env.dt)
    )

    rng, rng_reset = jax.random.split(rng)  # NOTE: rng_reset should never be changed.
    state_init = reset_env_jit(rng_reset)

    ## run diffusion

    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)
    Sigmas_cond = (
        (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
    )
    sigmas_cond = jnp.sqrt(Sigmas_cond)
    sigmas_cond = sigmas_cond.at[0].set(0.0)
    # print(f"init sigma = {sigmas[-1]:.2e}")

    mu_0T = jnp.zeros([args.Hsample, Nu])

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
        with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for t in pbar:
                carry_once = (t, rng, mu_0t)
                (t, rng, mu_0t), rew = reverse_once(carry_once, None)
                mu_0ts.append(mu_0t)
                # Update the progress bar's suffix to show the current reward
                pbar.set_postfix({'rew': f'{rew:.2e}'})
        return mu_0ts

    rng_exp, rng = jax.random.split(rng)
    mu_0ts = reverse(mu_0T, rng_exp)
    mu_0ts = jnp.array(mu_0ts)
    if args.render:
        path = f"{mbd.__path__[0]}/../results/{args.env_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        jnp.save(f"{path}/mu_0ts.npy", mu_0ts)
        webpage = render_us(state_init, mu_0ts[-1])
        with open(f"{path}/rollout.html", "w") as f:
            f.write(webpage)
    rew_final = eval_us(state_init, mu_0ts[-1]).mean()

    return rew_final
    


if __name__ == "__main__":
    run_diffusion(args=tyro.cli(Args))
