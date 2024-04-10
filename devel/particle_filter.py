import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# parameters
key = jax.random.PRNGKey(0)
N = 1024  # sampled trajectory number
H = 10  # horizon
map_scale = 0.5
# dt = 5.1 / H * map_scale
# dt = 10.1 / H * map_scale
dt = 11.1 / H * map_scale


# setup dynamics
def check_pass_wall(p1, p2, x1, x2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, x1, x2) != ccw(p2, x1, x2)) & (ccw(p1, p2, x1) != ccw(p1, p2, x2))


def f(x, u):
    u = jnp.clip(u, -1.0, 1.0)

    x_new = x + dt * u

    # check if the particle passes the wall
    pass_wall1 = check_pass_wall(
        jnp.array([-2.0, 1.0]) * map_scale, jnp.array([0.0, 1.0]) * map_scale, x, x_new
    )
    pass_wall2 = check_pass_wall(
        jnp.array([0.0, -1.0]) * map_scale, jnp.array([0.0, 1.0]) * map_scale, x, x_new
    )
    pass_wall3 = check_pass_wall(
        jnp.array([-2.0, -1.0]) * map_scale, jnp.array([0.0, -1.0]) * map_scale, x, x_new
    )
    pass_wall = pass_wall1 | pass_wall2 | pass_wall3

    x_new = jnp.where(pass_wall, x, x_new)

    return x_new


def cost(x):
    return jnp.sum((x - jnp.array([1.0, 0.0]) * map_scale) ** 2)


def plot_dyn(xs, ys, name="foo", xss=None):
    plt.figure()
    plt.plot(xs[:, 0], xs[:, 1], c="r", alpha=0.5)
    plt.scatter(
        [x[0] for x in ys],
        [x[1] for x in ys],
        label="observation",
        c=range(xs.shape[0]),
        cmap="Blues",
        marker="o",
        alpha=1.0,
        s=200,
    )
    plt.scatter(
        [x[0] for x in xs],
        [x[1] for x in xs],
        label="state",
        c=range(xs.shape[0]),
        cmap="Reds",
        marker="o",
        alpha=0.3,
    )
    if xss is not None:
        for i in range(xss.shape[0]):
            plt.plot(
                xss[i, :, 0],
                xss[i, :, 1],
                c="black",
                alpha=0.1,
                linewidth=1,
                linestyle="--",
            )
            # plt.scatter(
            #     [x[0] for x in xss[i]],
            #     [x[1] for x in xss[i]],
            #     c=range(xss[i].shape[0]),
            #     cmap="Greys",
            #     marker="o",
            #     alpha=0.3,
            # )
    # plot barrier with wide line
    plt.plot(
        [-2.0 * map_scale, 0.0, 0.0, -2.0 * map_scale],
        [-1.0 * map_scale, -1.0 * map_scale, 1.0 * map_scale, 1.0 * map_scale],
        "k",
        linewidth=10,
    )
    # set x, y limits
    plt.xlim(jnp.array([-5.0, 5.0])*map_scale)
    plt.ylim(jnp.array([-5.0, 5.0])*map_scale)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.legend()
    # save plot
    plt.savefig(f"../figure/{name}.png")
    plt.savefig(f"../figure/ww.png")
    plt.close()


# simulate the dynamics
def rollout_traj(x0, us):
    # rollout with jax.lax.scan
    def step(x, u):
        x_new = f(x, u)
        return x_new, x_new

    _, xs = jax.lax.scan(step, x0, us)
    return jnp.concatenate([x0[None, :], xs], axis=0)


# us = jnp.array([[1.0, 0.0] for _ in range(10)])
# xs = rollout_traj(jnp.array([-1.0, 0.0]), us)
# plot_dyn(xs, us)


# get the likelihood of the trajectory
def get_logpd(ys, xs, sigma):
    return -0.5 * jnp.mean((((ys - xs) / sigma)[1:] ** 2).sum(axis=1), axis=0)

def get_logpu(w_us, sigma):
    # NOTE: logpu is inverse of Gaussian, so it is positive
    return 0.5 * jnp.mean(((w_us / sigma) ** 2).sum(axis=1), axis=0)

def get_logpc(xs):
    # costs = jax.vmap(cost)(xs)
    # costs = costs.at[0].set(jnp.clip(costs[0], -(1.0/3.0)**2, 0.0) * 10.0)
    # costs = costs.at[-1].set(costs[-1] * 10.0)
    # costs = costs.at[0].set(costs[0]*10.0)
    # costs = costs.at[0].set((jnp.linalg.norm(xs[0] - jnp.array([-1.0, 0.0])) ** 2)*100.0)
    dist2goal = jnp.linalg.norm(xs - jnp.array([1.0, 0.0]) * map_scale, axis=1)
    close2goal_cost = jnp.clip(dist2goal/map_scale, 0.0, 1.0) ** 2
    logpc = -close2goal_cost[-1:].sum()*2.0-dist2goal[1:].mean()*1.0
    # get_x2g = lambda x: jnp.clip(jnp.linalg.norm(x - jnp.array([1.0, 0.0])), 0.0, 1.0) ** 2
    # xs2g = jax.vmap(get_x2g)(xs)
    # logpc = - xs2g.sum()
    # xf2g = jnp.linalg.norm(xs[-1] - jnp.array([1.0, 0.0])) ** 2
    # xf2g = jnp.clip(xf2g, 0.0, 1.0)
    # cost = - xf2g ** 2
    return logpc


def get_logp(ys, xs, sigma, pc_weight=1.0):
    logpd = get_logpd(ys, xs, sigma)
    logpc = get_logpc(xs)
    return logpc * pc_weight + logpd * 0.1

def sample_gmm(key, weights, means, sigmas):
    key, subkey = jax.random.split(key)
    idx = jax.random.categorical(subkey, jnp.log(weights))
    key, subkey = jax.random.split(key)
    return jax.random.normal(subkey, (H, 2)) * sigmas[idx] + means[idx]

# run MPPI
def mppi_traj(us_batch, key):
    # us_key, key = jax.random.split(key)
    # us_batch = jax.random.normal(us_key, (N, H, 2)) * 1.0 + us
    # us_batch = jnp.clip(us_batch, -1.0, 1.0)
    xs_batch = jax.vmap(rollout_traj, in_axes=(None, 0))(
        jnp.array([-1.0, 0.0])*map_scale, us_batch
    )
    logpc = jax.vmap(get_logpc)(xs_batch)*1.0
    w_unnorm = jnp.exp((logpc - jnp.max(logpc)))
    w = w_unnorm / jnp.sum(w_unnorm, axis=0)
    xs_batch_best = xs_batch[jnp.argmax(w)]

    sigmas = jnp.ones(N) * 0.1
    means = us_batch
    key_us, key = jax.random.split(key)
    us_batch = jax.vmap(sample_gmm, in_axes=(0, None, None, None))(jax.random.split(key_us, N), w, means, sigmas)
    us_batch = jnp.clip(us_batch, -1.0, 1.0)

    # us = jnp.sum(w[:, None, None] * us_batch, axis=0)
    # us_batch = jnp.clip(us + jax.random.normal(key, (H, 2)) * 1.0, -1.0, 1.0)

    return us_batch, key, xs_batch, xs_batch_best


us_key, key = jax.random.split(key)
us_batch = jax.random.normal(us_key, (N, H, 2)) * 1.0
us_batch = jnp.clip(us_batch, -1.0, 1.0)
# mppi_traj_jit = jax.jit(mppi_traj)
for i in range(10):
    us_batch, key, xs_batch, xs_mppi = mppi_traj(us_batch, key)
    xf_batch = xs_batch[:, -1]
    xf_reach_goal = jnp.linalg.norm(xf_batch/map_scale - jnp.array([1.0, 0.0]), axis=1) < 1.0
    # filter out xs_batch that reach the goal
    # xs_batch = xs_batch[xf_reach_goal]
    plot_dyn(xs_mppi, xs_mppi, "MPPI", xs_batch[xf_reach_goal])

# plt.figure()
def denoise_traj(ys, us, sigma, key):
    # filter for new trajectory
    # all_moved = jnp.zeros((N), dtype=bool)
    # xs_batch = jnp.zeros((N, H+1, 2))
    # while not all_moved.all():
    us_key, key = jax.random.split(key)
    us_batch = jnp.clip(us + jax.random.normal(us_key, (N, H, 2)) * sigma * 1.0, -1.0, 1.0)
    w_us = us_batch - us
    xs_batch_new = jax.vmap(rollout_traj, in_axes=(None, 0))(
        jnp.array([-1.0, 0.0]) * map_scale, us_batch
    )
    xs_batch = xs_batch_new
        # all_moved_new = jnp.all(jnp.linalg.norm(jnp.diff(xs_batch_new, axis=1)[:, :1], axis=2) > 1e-3, axis=1)
        # xs_batch = jnp.where(all_moved_new[:, None, None], xs_batch_new, xs_batch)
        # all_moved = all_moved_new | all_moved
        # print(all_moved)

    # rollout the trajectory assume start from the last state
    # us_inv_key, key = jax.random.split(key)
    # us_batch_inverse = (
    #     -jnp.flip(us, axis=0) + jax.random.normal(us_inv_key, (N, H, 2)) * sigma * 1.0
    # )
    # us_batch_inverse = jnp.clip(us_batch_inverse, -1.0, 1.0)
    # xs_batch_inverse = jax.vmap(rollout_traj, in_axes=(None, 0))(
    #     jnp.array([1.0, 0.0]) * map_scale, us_batch_inverse
    # )
    # xs_batch = jnp.concatenate([xs_batch, jnp.flip(xs_batch_inverse, axis=1)], axis=0)
    # us_batch = jnp.concatenate([us_batch, -jnp.flip(us_batch_inverse, axis=1)], axis=0)

    # logps = jax.vmap(get_logp, in_axes=(None, 0, None, None))(ys, xs_batch, sigma, 0.1)
    logpd = jax.vmap(get_logpd, in_axes=(None, 0, None))(ys, xs_batch, sigma)
    logpc = jax.vmap(get_logpc)(xs_batch)
    logpu = jax.vmap(get_logpu, in_axes=(0, None))(w_us, sigma)
    # logps = logpd*15.0 + logpc*0.5
    # logps = logpd*6.0 + logpc*6.0
    logps = logpd*8.0 + logpc*8.0 + logpu*0.0

    w_unnorm = jnp.exp((logps - jnp.max(logps)))
    w = w_unnorm / jnp.sum(w_unnorm, axis=0)
    # plt.hist(w, bins=20)
    # plt.savefig("../figure/w.png")
    # plt.cla()

    # print(f"logpd: {logpd.mean():.2f} \pm {logpd.std():.2f} logpc: {logpc.mean():.2f} \pm {logpc.std():.2f}")

    # plot w with histogram
    # plt.hist(w, bins=20)
    # plt.show()
    # plt.cla()

    us_new = jnp.sum(w[:, None, None] * us_batch, axis=0)
    xs_new = jnp.sum(w[:, None, None] * xs_batch, axis=0)
    return xs_new, us_new, key, jnp.concat([xs_batch[:4], xs_batch[-4:]], axis=0)


us_key, key = jax.random.split(key)
us = jax.random.normal(us_key, (H, 2)) * 1.0
ys_key, key = jax.random.split(key)
xs_guess = jnp.linspace(-1.0, 1.0, H + 1)[:, None] * jnp.array([1.0, 0.0]) * map_scale
ys = jax.random.normal(ys_key, (H + 1, 2)) * 1.0 + xs_guess
# ys = xs_guess + jnp.array([0.0, 1.5])
ys = ys.at[0, :2].set(jnp.array([-1.0, 0.0]) * map_scale)
ys = ys.at[-1, :2].set(jnp.array([1.0, 0.0]) * map_scale)
# ys = ys.at[1, :2].set(jnp.array([-3.0, 1.3]))
# ys = ys.at[2, :2].set(jnp.array([0.4, 1.3]))
# ys = ys.at[1, :2].set(jnp.array([-2.0, 2.0]) * map_scale*0.0)
# ys = ys.at[2, :2].set(jnp.array([0.0, 2.0]) * map_scale*0.0)
# ys = ys.at[1, :2].set(jnp.array([0.5, 0.0]))
# ys = ys.at[2, :2].set(jnp.array([0.5, 0.0]))
# xs_ref = jnp.array([[-1.0, 0.0], [-2.0, 2.0], [0.0, 2.0], [1.0, 0.0]]) * map_scale
# set us to ys
us = jnp.diff(ys, axis=0) / dt
us = jnp.clip(us, -1.0, 1.0)
# us = jnp.zeros((H, 2))

T = 500

def get_alpha_bar(t):
    # x = (t / T - 0.5) * 8.0
    x = (t/T)*6.0 - 3.0 # from 2.0 to -5.0
    return jax.nn.sigmoid(-x)


ts = jnp.arange(T + 1)
alpha_bars = get_alpha_bar(ts)
alphas = alpha_bars / jnp.roll(alpha_bars, 1)
denoise_traj_jit = jax.jit(denoise_traj)
# for (i, var) in enumerate(np.arange(0.1, 0.0, -var_step)):
for i in range(T, 0, -1):
    alpha_bar_prev = alpha_bars[i - 1]
    alpha_bar = alpha_bars[i]
    alpha = alphas[i]

    var = 1 - alpha_bar
    var_cond = (1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)

    sigma = jnp.sqrt(var)
    # x|yi
    xs, us, key, xs_batch = denoise_traj_jit(ys, us, sigma, key)
    # us = jnp.diff(ys, axis=0) / dt
    # us = jnp.clip(us, -1.0, 1.0)
    if i % 20 == 19:
        plot_dyn(xs, ys, "denoise", xs_batch)

    # if var <= var_step:
    #     sigma_ys = jnp.sqrt(var_step)
    # else:
    #     # sigma_ys = jnp.sqrt(1.0 / (1.0 / var_step + 1.0 / (var - var_step)))
    #     sigma_ys = jnp.sqrt(var_step)
    # yi-1|yi
    ys_key, key = jax.random.split(key)
    # ys = xs + (ys-xs)*(var-var_step)/(var) + jax.random.normal(ys_key, (H+1, 2)) * sigma_ys
    # xs = xs_ref
    ys = (
        jnp.sqrt(alpha) * (1 - alpha_bar_prev) * ys
        + jnp.sqrt(alpha_bar_prev) * (1 - alpha) * xs
    ) / (1 - alpha_bar) + jax.random.normal(ys_key, (H + 1, 2)) * jnp.sqrt(var_cond)
    # print(jnp.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar), jnp.sqrt(alpha_bar_prev) * (1 - alpha) / (1 - alpha_bar), jnp.sqrt(var_cond))
    ys = ys.at[0, :2].set(jnp.array([-1.0, 0.0])*map_scale)
    ys = ys.at[-1, :2].set(jnp.array([1.0, 0.0])*map_scale)
    # key_yf, key = jax.random.split(key)
    # ys = ys.at[-1, :2].set((jnp.array([1.0, 0.0])+jax.random.normal(key_yf, (2,))*sigma)*map_scale)