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
import pickle

# config.update("jax_enable_x64", True) # NOTE: this is important for simulating long horizon open loop control

## global config

use_data = False
init_data = False

## setup env

env_name = "humanoidtrack"
backend = "generalized"
if env_name in ["hopper", "walker2d"]:
    substeps = 10
elif env_name in ["humanoid", "humanoidstandup"]:
    substeps = 2
else:
    substeps = 1
if env_name == "pushT":
    from pushT import PushT

    env = PushT()
elif env_name == "humanoidtrack":
    from humanoidtrack import HumanoidTrack

    env = HumanoidTrack()
    terminal_cost_scale = 10.0
    
    # load demostration from ../devel/xs_ref_dict.pkl
    with open("../devel/run_ref_dict.pkl", "rb") as f:
        xs_demo_dict = pickle.load(f)

    # plt.plot(left_shin_demo, label="left shin demo")
    # plt.plot(right_shin_demo, label="right shin demo")
    # plt.savefig("../figure/gait.png")

    def eval_xs(state, us):
        def step(state, data):
            u, t = data
            state = step_env(state, u)
            value = 0.0
            for name, idx in env.track_body_idx.items():
                x = state.pipeline_state.x.pos[idx]
                value -= (x - xs_demo_dict[name][t]) ** 2
            return state, value

        _, value_xs = jax.lax.scan(step, state, (us, jnp.arange(us.shape[0]) + 1))
        return value_xs.mean()

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
Nsample = 8192 
Hsample = 50
Ndiffuse = 200
temp_sample = 0.1
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

mu_0T = jnp.zeros([Nexp, Hsample, Nu])


# evaluate the diffused uss
@jax.jit
def eval_us(state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, state.reward

    _, rews = jax.lax.scan(step, state, us)
    rews = rews.at[-1].set(rews[-1] * terminal_cost_scale)
    return rews

def set_ref_body_pos(env, pipeline_state, xs_demo_dict, t):
    for i, (name, idx) in enumerate(env.ref_body_idx.items()):
        pipeline_state = pipeline_state.replace(
            x=pipeline_state.x.replace(
                pos=pipeline_state.x.pos.at[idx].set(xs_demo_dict[name[:-4]][t])
            )
        )
    return pipeline_state

def render_us(state, us):
    rollout = []
    rew_sum = 0.0
    for t in range(Hsample):
        for j in range(substeps):
            pipeline_state = state.pipeline_state
            pipeline_state = set_ref_body_pos(env, pipeline_state, xs_demo_dict, t)
            rollout.append(pipeline_state)
            state = step_env_jit(state, us[t])
            rew_sum += state.reward
    rew_mean = rew_sum / (Hsample * substeps)
    webpage = html.render(env.sys.replace(dt=env.dt), rollout)
    print(f"evaluated reward mean: {rew_mean:.2e}")
    with open(f"{path}/rollout.html", "w") as f:
        f.write(webpage)


@jax.jit
def reverse_once(carry, unused):
    t, rng, mu_0t = carry

    # sample from q_i
    rng, Y0s_rng = jax.random.split(rng)
    eps_u = jax.random.normal(Y0s_rng, (Nsample, Hsample, Nu))
    Y0s = eps_u * sigmas[t] + mu_0t
    Y0s = jnp.clip(Y0s, -1.0, 1.0)

    # esitimate mu_0tm1
    # rews = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0s).mean(axis=-1)
    # jax.debug.print("rews={x} \pm {y}", x=rews.mean(), y=rews.std())
    # logp0 = (rews - rews.mean()) / rews.std() / temp_sample
    # weights = jax.nn.softmax(logp0)
    # mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)  # NOTE: update only with reward

    # esitimate mu_0tm1
    # rews = jax.vmap(eval_us, in_axes=(None, 0))(state_init, Y0s).mean(axis=-1)
    # rews_normed = (rews - rews.mean()) / rews.std()
    value_xs = jax.vmap(eval_xs, in_axes=(None, 0))(state_init, Y0s)
    rews = jnp.zeros(Nsample)
    rews_normed = jnp.zeros(Nsample)
    logpJ = rews_normed / temp_sample
    logpdemo = ((value_xs - jnp.mean(value_xs))/jnp.std(value_xs) + rews_normed) / temp_sample
    # jax.debug.print("rews={x} \pm {y}", x=rews.mean(), y=rews.std())
    jax.debug.print("value_xs={x} \pm {y}", x=value_xs.mean(), y=value_xs.std())
    # logp0 = jnp.concat([logpJ, logpdemo], axis=0)
    # weights = jax.nn.softmax(logp0)
    # mu_0tm1 = jnp.einsum("n,nij->ij", weights, jnp.concatenate([Y0s, Y0s], axis=0))
    weights = jax.nn.softmax(logpdemo)
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)

    return (t - 1, rng, mu_0tm1), rews.mean()


# run reverse
def reverse(mu_0T, rng):
    carry_once = (Ndiffuse - 1, rng, mu_0T)
    (_, rng, mu_0T), rew = jax.lax.scan(reverse_once, carry_once, None, Ndiffuse)
    return mu_0T, rew


rng_exp = jax.random.split(rng, Nexp)
mu_0, rew_exp = jax.vmap(reverse)(mu_0T, rng_exp)
rew_eval = jax.vmap(eval_us, in_axes=(None, 0))(state_init, mu_0).mean(axis=-1)
print(f"rews mean: {rew_eval.mean():.2e} std: {rew_eval.std():.2e}")

render_us(state_init, mu_0[jnp.argmax(rew_eval)])