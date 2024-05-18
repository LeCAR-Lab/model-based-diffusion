import functools
import os
import jax
from jax import numpy as jnp
from jax import config
from dataclasses import dataclass
import tyro
from tqdm import tqdm
from matplotlib import pyplot as plt

import mbd

# NOTE: enable this if you want higher precision
# config.update("jax_enable_x64", True)


## load config
@dataclass
class Args:
    # exp
    seed: int = 0
    disable_recommended_params: bool = False
    not_render: bool = False
    # env
    env_name: str = (
        "ant"  # "humanoidstandup", "ant", "halfcheetah", "hopper", "walker2d", "car2d"
    )
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 50  # horizon
    Ndiffuse: int = 100  # number of diffusion steps
    temp_sample: float = 0.1  # temperature for sampling
    beta0: float = 1e-4  # initial beta
    betaT: float = 1e-2  # final beta
    enable_demo: bool = False


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
        "pushT": 0.2,
    }
    Ndiffuse_recommend = {
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
        args.Ndiffuse = Ndiffuse_recommend.get(args.env_name, args.Ndiffuse)
        args.Nsample = Nsample_recommend.get(args.env_name, args.Nsample)
        args.Hsample = Hsample_recommend.get(args.env_name, args.Hsample)
        print(f"override temp_sample to {args.temp_sample}")
    env = mbd.envs.get_env(args.env_name)
    Nx = env.observation_size
    Nu = env.action_size
    # env functions
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    # eval_us = jax.jit(functools.partial(mbd.utils.eval_us, step_env_jit))
    rollout_us = jax.jit(functools.partial(mbd.utils.rollout_us, step_env_jit))

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
    print(f"init sigma = {sigmas[-1]:.2e}")

    mu_0T = jnp.zeros([args.Hsample, Nu])

    @jax.jit
    def reverse_once(carry, unused):
        t, rng, mu_0t = carry
        # mu_0t = mu_t / jnp.sqrt(alphas_bar[t])

        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_u = jax.random.normal(Y0s_rng, (args.Nsample, args.Hsample, Nu))
        Y0s = eps_u * sigmas[t] + mu_0t
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # esitimate mu_0tm1
        rewss, qs = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=-1)
        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        rew_mean = rews.mean()
        logp0 = (rews - rew_mean) / rew_std / args.temp_sample

        # evalulate demo
        if args.enable_demo:
            xref_logpds = jax.vmap(env.eval_xref_logpd)(qs)
            xref_logpds = xref_logpds - xref_logpds.max()
            logpdemo = (
                (xref_logpds + env.rew_xref - rew_mean) / rew_std / args.temp_sample
            )
            demo_mask = logpdemo > logp0
            logp0 = jnp.where(demo_mask, logpdemo, logp0)
            logp0 = (logp0 - logp0.mean()) / logp0.std() / args.temp_sample

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
                pbar.set_postfix({"rew": f"{rew:.2e}"})
        return jnp.array(mu_0ts)

    rng_exp, rng = jax.random.split(rng)
    mu_0ts = reverse(mu_0T, rng_exp)
    if not args.not_render:
        path = f"{mbd.__path__[0]}/../results/{args.env_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        jnp.save(f"{path}/mu_0ts.npy", mu_0ts)
        if args.env_name == "car2d":
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            # rollout
            xs = jnp.array([state_init.pipeline_state])
            state = state_init
            for t in range(mu_0ts.shape[1]):
                state = step_env_jit(state, mu_0ts[-1, t])
                xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
            env.render(ax, xs)
            if args.enable_demo:
                ax.plot(env.xref[:, 0], env.xref[:, 1], "g--", label="RRT path")
            ax.legend()
            plt.savefig(f"{path}/rollout.png")
        else:
            render_us = functools.partial(
                mbd.utils.render_us, step_env_jit, env.sys.replace(dt=env.dt)
            )
            webpage = render_us(state_init, mu_0ts[-1])
            with open(f"{path}/rollout.html", "w") as f:
                f.write(webpage)
    rewss_final, _ = rollout_us(state_init, mu_0ts[-1])
    rew_final = rewss_final.mean()

    return rew_final


if __name__ == "__main__":
    rew_final = run_diffusion(args=tyro.cli(Args))
    print(f"final reward = {rew_final:.2e}")
