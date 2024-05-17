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

use_data = False
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
rng = jax.random.PRNGKey(seed=1)
rng, rng_reset = jax.random.split(rng)  # NOTE: rng_reset should never be changed.
state_init = reset_env(rng_reset)
path = f"../figure/{env_name}/{backend}"
if not os.path.exists(path):
    os.makedirs(path)

## run diffusion

Nexp = 1
Nsample = 1024
Hsample = 50
Ndiffuse = 100
temp_sample = 0.5
beta0 = 1e-4
betaT = 1e-2
betas = jnp.linspace(beta0, betaT, Ndiffuse)
# betas = jnp.exp(jnp.linspace(jnp.log(beta0), jnp.log(betaT), Ndiffuse))
alphas = 1.0 - betas
alphas_bar = jnp.cumprod(alphas)
sigmas = jnp.sqrt(1 - alphas_bar)
Sigmas_cond = (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
sigmas_cond = jnp.sqrt(Sigmas_cond)
sigmas_cond = sigmas_cond.at[0].set(0.0)
print(f"init sigma = {sigmas[-1]:.2e}")
# sigma0 = 1e-2
# sigmaT = 0.9
# sigmas = jnp.exp(jnp.linspace(jnp.log(sigma0), jnp.log(sigmaT), Ndiffuse))
# alphas_bar = 1 - sigmas**2
# betas = 1 - alphas_bar
# alphas = 1 - betas
# Sigmas_cond = (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
# Sigmas_cond = Sigmas_cond.at[0].set(0.0)
# sigmas_cond = jnp.sqrt(Sigmas_cond)
plt.plot(sigmas, label = "sigma")
plt.plot(alphas_bar, label = "alpha_bar")
plt.plot(betas, label = "beta")
plt.plot(sigmas_cond, label = "sigma_cond")
plt.legend()
plt.savefig(f"{path}/foo.png")
# exit() 

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
    rew_Y0_hat = eval_us(state_init, Y0_hat).mean()
    rew_Yt = eval_us(state_init, Yt/jnp.sqrt(alphas_bar[t])).mean()
    k_Y0_hat = jax.nn.softmax(jnp.array([rew_Y0_hat, rew_Yt]) / temp_sample)[0]
    N_Y0_hat = jnp.int32(Nsample * k_Y0_hat)
    sample_Y0_hat_mask = jnp.arange(Nsample) < N_Y0_hat
    Y0_mean1 = jnp.repeat(Y0_hat[None], Nsample, axis=0)
    Y0_mean2 = jnp.repeat((Yt/jnp.sqrt(alphas_bar[t]))[None], Nsample, axis=0)
    Y0s_mean = jnp.where(sample_Y0_hat_mask[:, None, None], Y0_mean1, Y0_mean2)
    Y0s = eps_u * sigmas[t] + Y0s_mean
    eps_u_Y0_hat = (Y0s - Y0_mean1) / sigmas[t]
    eps_u_Yt = (Y0s - Y0_mean2) / sigmas[t]
    logpdss_Y0_1 = -0.5 * jnp.mean(eps_u_Y0_hat**2, axis=-1)
    logpdss_Y0_2 = -0.5 * jnp.mean(eps_u_Yt**2, axis=-1)
    logpdss_Y0 = k_Y0_hat * logpdss_Y0_1 + (1 - k_Y0_hat) * logpdss_Y0_2
    jax.debug.print("======")
    jax.debug.print("k_Y0_hat = {x} \n rew_Y0_hat={y} \n rew_Yt={z}", 
                    x=k_Y0_hat, y=rew_Y0_hat, z=rew_Yt)

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
    eps_Y = jnp.clip((Y0s * jnp.sqrt(alphas_bar[t]) - Yt) / sigmas[t], -2.0, 2.0)
    logpdss_Yt = -0.5 * jnp.mean(eps_Y**2, axis=-1)
    logpdss = logpdss_Yt - logpdss_Y0
    logpds = logpdss.mean(axis=-1)
    rews = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0s).mean(axis=-1)
    rews_normed = (rews - rews.mean()) / rews.std()
    logweight = rews_normed + logpds
    weights = jax.nn.softmax(logweight / temp_sample)
    weights_rew = jax.nn.softmax(rews_normed / temp_sample)
    Y0_bar = jnp.einsum("n,nij->ij", weights, Y0s)
    # Get new Y0_hat
    Y0_hat_new = jnp.einsum("n,nij->ij", weights_rew, Y0s) # NOTE: update only with reward

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
    score = alphas_bar[t] / (1 - alphas_bar[t]) * (Y0_bar - Yt / jnp.sqrt(alphas_bar[t]))

    # calculate Ytm1
    rng, Ytm1_rng = jax.random.split(rng)
    eps_Ytm1 = jax.random.normal(Ytm1_rng, (Hsample, Nu))
    Ytm1 = (
        1 / jnp.sqrt(1.0 - betas[t]) * (Yt + 0.5 * betas[t] * score)
        + 1.0*jnp.sqrt(betas[t]) * eps_Ytm1
    )

    return (t - 1, rng, Y0_hat_new, Ytm1), rews.mean()


# run reverse
def reverse(Y0_hat, Yt, rng):
    carry_once = (Ndiffuse - 1, rng, Y0_hat, Yt)
    (t0, rng, Y0_hat, Y0), rew = jax.lax.scan(reverse_once, carry_once, None, Ndiffuse)
    # for i in range(Ndiffuse):
    #     carry_once, rew = reverse_once(carry_once, None)
    # (tT, rng, Y0_hat, Y0), rew = carry_once
    return Y0_hat, Y0, rew


rng_exp = jax.random.split(rng, Nexp)
Y0_hat_exp, Y0_exp, rew_exp = jax.vmap(reverse)(Y0_hat_exp, Yt_exp, rng_exp)
rew_eval = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0_hat_exp).mean(axis=-1)
print(f"rews mean: {rew_eval.mean():.2e} std: {rew_eval.std():.2e}")

render_us(state_init, Y0_hat_exp[jnp.argmax(rew_eval)])