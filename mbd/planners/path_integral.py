import functools
import jax
from jax import numpy as jnp
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
    update_method: str = "mppi"  # mppi, cma-es, cem
    # env
    env_name: str = (
        "ant"  # "humanoidstandup", "ant", "halfcheetah", "hopper", "walker2d"
    )
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 50  # horizon
    Nrefine: int = 100  # number of repeat steps
    temp_sample: float = 0.1  # temperature for sampling


@jax.jit
def softmax_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma


@jax.jit
def cma_es_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)
    Yerr = Y0s - mu_0t
    sigma = jnp.sqrt(jnp.einsum("n,nij->ij", weights, Yerr**2)).mean() * sigma
    sigma = jnp.maximum(sigma, 1e-3)
    return mu_0tm1, sigma


@jax.jit
def cem_update(weights, Y0s, sigma, mu_0t):
    idx = jnp.argsort(weights)[::-1][:10]
    mu_0tm1 = jnp.mean(Y0s[idx], axis=0)
    return mu_0tm1, sigma


def run_path_integral(args: Args):

    rng = jax.random.PRNGKey(seed=args.seed)

    update_fn = {
        "mppi": softmax_update,
        "cma-es": cma_es_update,
        "cem": cem_update,
    }[args.update_method]

    ## setup env

    # recommended temperature for envs
    temp_recommend = {
        "ant": 0.1,
        "halfcheetah": 0.4,
        "hopper": 0.1,
        "humanoidstandup": 0.1,
        "humanoidrun": 0.1,
        "walker2d": 0.1,
        "pushT": 0.2,
    }
    Nrefine_recommend = {
        "pushT": 200,
        "humanoidrun": 300,
    }
    Nsample_recommend = {
        "humanoidrun": 8192,
    }
    Hsample_recommend = {
        "pushT": 40,
    }
    if not args.disable_recommended_params:
        args.temp_sample = temp_recommend.get(args.env_name, args.temp_sample)
        args.Nrefine = Nrefine_recommend.get(args.env_name, args.Nrefine)
        args.Nsample = Nsample_recommend.get(args.env_name, args.Nsample)
        args.Hsample = Hsample_recommend.get(args.env_name, args.Hsample)
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

    ## run path interal

    mu_0T = jnp.zeros([args.Hsample, Nu])

    @jax.jit
    def update_once(carry, unused):
        t, rng, mu_0t, sigma = carry

        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_u = jax.random.normal(Y0s_rng, (args.Nsample, args.Hsample, Nu)) * sigma
        Y0s = eps_u + mu_0t
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # esitimate mu_0tm1
        rews = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0s).mean(axis=-1)
        logp0 = (rews - rews.mean()) / rews.std() / args.temp_sample
        weights = jax.nn.softmax(logp0)
        mu_0tm1, sigma = update_fn(weights, Y0s, sigma, mu_0t)

        return (t - 1, rng, mu_0tm1, sigma), rews.mean()

    # run reverse
    def update(mu_0T, rng):
        sigma = 1.0
        mu_0t = mu_0T
        mu_0ts = []
        with tqdm(range(args.Nrefine - 1, 0, -1), desc="Path Integrating") as pbar:
            for t in pbar:
                carry_once = (t, rng, mu_0t, sigma)
                (t, rng, mu_0t, sigma), rew = update_once(carry_once, None)
                mu_0ts.append(mu_0t)
                # Update the progress bar's suffix to show the current reward
                pbar.set_postfix({"rew": f"{rew:.2e}"})
        return mu_0ts

    rng_exp, rng = jax.random.split(rng)
    mu_0ts = update(mu_0T, rng_exp)
    mu_0ts = jnp.array(mu_0ts)
    rew_final = eval_us(state_init, mu_0ts[-1]).mean()

    return rew_final


if __name__ == "__main__":
    rew = run_path_integral(args=tyro.cli(Args))
    print(f"rew: {rew:.2e}")
