import functools
from datetime import datetime
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model, html
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt

## setup env

env_name = "ant"
backend = "positional"
env = envs.get_environment(env_name=env_name, backend=backend)
rng = jax.random.PRNGKey(seed=0)
rng, rng_reset = jax.random.split(rng)
state = jax.jit(env.reset)(rng=rng_reset)

"""
## train
train_fn = {
    "ant": functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=10,
        reward_scaling=10,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=4096,
        batch_size=2048,
        seed=1,
    ),
}[env_name]

max_y = {"ant": 8000}[env_name]
min_y = {"ant": 0}[env_name]

fig, ax = plt.subplots()
xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics["eval/episode_reward"])
    ax.clear()
    ax.set_xlim([0, train_fn.keywords["num_timesteps"]])
    ax.set_ylim([min_y, max_y])
    ax.set_xlabel("# environment steps")
    ax.set_ylabel("reward per episode")
    ax.plot(xdata, ydata)
    plt.pause(0.01)


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

model.save_params(f"../figure/{env_name}/{backend}/params", params)

"""

## load model
jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)


load_backend = "spring"
params = model.load_params(f"../figure/{env_name}/{load_backend}/params")
normalize = running_statistics.normalize
ppo_network = ppo_networks.make_ppo_networks(
    state.obs.shape[-1], env.action_size, preprocess_observations_fn=normalize
)
make_policy = ppo_networks.make_inference_fn(ppo_network)
inference_fn = make_policy(params)
jit_inference_fn = jax.jit(inference_fn)


test_env = envs.get_environment(env_name=env_name, backend=load_backend)
jit_test_env_step = jax.jit(test_env.step)

state = jit_env_reset(rng=rng_reset)
test_state = jax.jit(test_env.reset)(rng=rng_reset)
assert jnp.all(state.obs == test_state.obs), "reset state is different"

reward_sum = 0.0
test_reward_sum = 0.0
Nrollout = 1024 * 64
Hrollout = 100
us_policy = jnp.zeros([Hrollout, env.action_size])
rollout = []
rollout_test = []
for i in range(Hrollout):
    rollout.append(state.pipeline_state)
    rollout_test.append(test_state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)

    test_act, _ = jit_inference_fn(test_state.obs, act_rng)
    test_state = jit_test_env_step(test_state, test_act)

    us_policy = us_policy.at[i].set(test_act)

    reward_sum += state.reward
    test_reward_sum += test_state.reward

state = jit_env_reset(rng=rng_reset)
rollout_openloop = []
openloop_reward_sum = 0.0
for i in range(Hrollout):
    rollout_openloop.append(state.pipeline_state)
    state = jit_env_step(state, us_policy[i])
    openloop_reward_sum += state.reward

print(f"reward_sum (diff backend): {reward_sum/Hrollout:.2f}")
print(f"test_reward_sum (same backend): {test_reward_sum/Hrollout:.2f}")
print(f"openloop_reward_sum: {openloop_reward_sum/Hrollout:.2f}")

# webpage = html.render(env.sys.replace(dt=env.dt), rollout)
# # # save it to a file
# with open(f"../figure/{env_name}/{load_backend}/{backend}_RL_render.html", "w") as f:
#     f.write(webpage)
# webpage_test = html.render(env.sys.replace(dt=env.dt), rollout_test)
# with open(
#     f"../figure/{env_name}/{load_backend}/{load_backend}_RL_render.html", "w"
# ) as f:
#     f.write(webpage_test)
webpage_openloop = html.render(env.sys.replace(dt=env.dt), rollout_openloop)
with open(
    f"../figure/{env_name}/{load_backend}/{backend}_openloop_render.html", "w"
) as f:
    f.write(webpage_openloop)

"""

## Data collection

@jax.jit
def rollout_env(state, rng):
    def step(carry, unused):
        state, rng = carry
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)
        return (state, rng), (act, state.reward)

    _, (us, rews) = jax.lax.scan(step, (state, rng), None, Hrollout)
    return us, rews


rng = jax.random.PRNGKey(seed=0)
reset_state = jit_env_reset(rng=rng)
u_rng, rng = jax.random.split(rng)
uss, rews = jax.vmap(rollout_env, in_axes=(None, 0))(
    reset_state, jax.random.split(u_rng, Nrollout)
)
# save it to a file
print(f"collected uss with shape {uss.shape}")
print(f"collected rews = {rews.mean():.2e}")
jnp.save(f"../figure/{env_name}/{backend}/uss.npy", uss)


"""

## run MPPI
Nmppi = 1024
Nnode = 100
Hnode = 1
Hmppi = (Nnode - 1) * Hnode + 1
nx = env.observation_size
nu = env.action_size
temp_mppi = 0.5  # 0.1
us_node = jnp.zeros([Nnode, nu])
# us_node = us_policy[::Hnode]
y_rng, rng = jax.random.split(rng)
# yus_node = us_node + jax.random.normal(y_rng, us_node.shape) * sigmas[0]
yus_node = us_node


# evaluate the diffused uss
@jax.jit
def eval_us(state, us):
    def step(state, u):
        state = jit_env_step(state, u)
        return state, state.reward

    _, rews = jax.lax.scan(step, state, us)
    return rews


"""
uss = jnp.load(f"../figure/{env_name}/{backend}/uss_diffused.npy")
state = jit_env_reset(rng=jax.random.PRNGKey(seed=0))
rews = eval_us(state, uss[2])
print(f"rews mean={rews.mean()}")
"""

# rollout policy for initial u sequence
mppi_rng, rng = jax.random.split(rng)


def linear_interpolation_single(x0, x1):
    return jnp.linspace(x0, x1, Hnode + 1)[:-1]


@jax.jit
def linear_interpolation(us_node):
    us_node0 = us_node[:-1]
    us_node1 = us_node[1:]
    us_wolast = jax.vmap(linear_interpolation_single, in_axes=(0, 0))(
        us_node0, us_node1
    )
    us_wolast = jnp.reshape(us_wolast, (Hmppi - 1, nu))
    us = jnp.concatenate([us_wolast, us_node[-1:]], axis=0)
    return us


fig, axes = plt.subplots(1, 2)
state = jit_env_reset(rng_reset)
reward_sum = 0.0
Ndiffuse = 100

# def cosine_schedule(num_timesteps, s=0.008):
#     def f(t):
#         return jnp.cos((t / num_timesteps + s) / (1 + s) * 0.5 * jnp.pi) ** 2

#     x = jnp.linspace(0, num_timesteps, num_timesteps + 1)
#     alphas_cumprod = f(x) / f(jnp.array([0]))
#     betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
#     betas = jnp.clip(betas, 0.0001, 0.999)
#     return betas

betas = jnp.linspace(1e-4, 1e-2, Ndiffuse)
alphas = 1.0 - betas
alpha_bar_T = jnp.prod(alphas)
print(f"init sigma = {jnp.sqrt(1 - alpha_bar_T):.2e}")

for i in range(Ndiffuse, 0, -1):
    # sigma = sigmas[i]
    # sigma_prev = sigmas[i - 1]
    # alpha_bar = 1 - sigma**2
    # alpha_bar_prev = 1 - sigma_prev**2
    # alpha = alpha_bar_prev / alpha_bar
    alpha_t = alphas[i]
    alpha_tm1 = alphas[i - 1]  # init sigma = 0.2
    alpha_bar_t = jnp.prod(alphas[:i])
    alpha_bar_tm1 = jnp.prod(alphas[: i - 1])
    sigma_t = jnp.sqrt(1 - alpha_bar_t)
    sigma_tm1 = jnp.sqrt(1 - alpha_bar_tm1)
    Sigma_q_t = (1 - alpha_t) * (1 - jnp.sqrt(alpha_bar_tm1)) / (1.0 - alpha_bar_t)
    sigma_q_t = jnp.sqrt(Sigma_q_t)
    # print(f"sigma={sigma:.2e} alpha={alpha:.2e} alpha_bar={alpha_bar:.2e}")

    # sampling from p(yi)
    # rng, mppi_rng = jax.random.split(rng)
    # eps_u = jax.random.normal(mppi_rng, (Nmppi, Nnode, nu))
    # uss_node = jnp.sqrt(1 / alpha_bar_t - 1.0) * eps_u + yus_node / jnp.sqrt(
    #     alpha_bar_t
    # )
    # uss_node = jnp.clip(uss_node, -1.0, 1.0)
    # uss = jax.vmap(linear_interpolation, in_axes=(0))(uss_node)

    # sampling from q(y0)
    us_node_repeat = jnp.repeat(us_node[None, ...], Nmppi, axis=0)
    n_policy = int((i / Ndiffuse) * Nmppi)
    us_node_repeat = us_node_repeat.at[:n_policy].set(us_policy)
    rng, mppi_rng = jax.random.split(rng)
    eps_u = jax.random.normal(mppi_rng, (Nmppi, Nnode, nu))
    uss_node = eps_u * jnp.sqrt(1 / alpha_bar_t - 1.0) + us_node_repeat
    uss_node = jnp.clip(uss_node, -1.0, 1.0)
    uss = jax.vmap(linear_interpolation, in_axes=(0))(uss_node)

    rewss = jax.vmap(eval_us, in_axes=(None, 0))(state, uss)
    rews = jnp.mean(rewss, axis=-1)
    # normalize reward
    rews_normed = (rews - rews.mean()) / rews.std()

    # logp for sampling from p(yi)
    # logweight = rews_normed

    # logpd for sampling from q(y0)
    eps = (uss_node * jnp.sqrt(alpha_bar_t) - yus_node) / sigma_t
    logpdss = -0.5 * jnp.mean(eps**2, axis=-1) + 0.5 * jnp.mean(eps_u**2, axis=-1)
    logpds = jnp.mean(logpdss, axis=-1)
    logpds = jnp.clip(logpds - logpds.max(), -1.0, 0.0)
    # jax.debug.print("logpds mean={x:.2e} pm {y:.2e}", x=logpds.mean(), y=logpds.std())
    logweight = rews_normed + 0.0 * logpds

    weights = jax.nn.softmax(logweight / temp_mppi)
    # us = jnp.einsum("n,nij->ij", weights, uss)
    # for j in range(Hnode):
    #     state = jit_env_step(state, us[j])
    #     rollout.append(state.pipeline_state)
    #     reward_sum += state.reward
    us_node = jnp.einsum("n,nij->ij", weights, uss_node)
    us_node_best = uss_node[rews.argmax()]

    ys_rng, rng = jax.random.split(rng)
    ky = jnp.sqrt(alpha_t) * (1 - alpha_bar_tm1) / (1 - alpha_bar_t)
    kx = jnp.sqrt(alpha_bar_tm1) * (1 - alpha_t) / (1 - alpha_bar_t)
    score_model = (ky - 1) * yus_node + kx * us_node
    score_data = (ky - 1) * yus_node + kx * us_policy
    # jax.debug.print(f"ky={ky:.2f} kx={kx:.2f}")

    # k_data = i / Ndiffuse
    k_data = 0.0  # without data
    k_model = 1 - k_data
    yus_node = (
        yus_node
        + k_model * score_model
        + k_data * score_data
        + jax.random.normal(ys_rng, (Nnode, nu)) * sigma_q_t
    )

    # axes[0].cla()
    # axes[0].plot(yus_node[:, 0])
    # axes[0].plot(us_policy[:, 0], "--")
    # axes[0].set_ylim([-1, 1])
    # plt.pause(0.01)

    print(f"=================== {i} ===================")
    print(f"sigma_t = {sigma_t:.2e}")
    print(f"rew={rews.mean():.2f} pm {rews.std():.2f} max={rews.max():.2f}")
    # print(f"weight={weights.mean():.2f} pm {weights.std():.2f}")
    # plot histogram
    # axes[0].cla()
    # axes[0].hist(rews, bins=100)
    # axes[1].cla()
    # axes[1].hist(weights, bins=20)
    # axes[1].set_ylim([0, 100])
    # plt.pause(0.01)
assert jnp.allclose(state.obs, jit_env_reset(rng_reset).obs), "state is changed"

us = linear_interpolation(us_node_best)
rollout = []
state = jit_env_reset(rng=rng_reset)
reward_sum = 0.0
for i in range(Hmppi):
    rollout.append(state.pipeline_state)
    state = jit_env_step(state, us[i])
    reward_sum += state.reward

print(f"reward mean: {reward_sum/Hmppi}")
webpage = html.render(env.sys.replace(dt=env.dt), rollout)
with open(f"../figure/{env_name}/{backend}/mppi_render.html", "w") as f:
    f.write(webpage)
