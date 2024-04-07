import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# parameters
key = jax.random.PRNGKey(1)
Nx = 128  # sampled trajectory number
Ny = 1 # observation number
H = 3  # horizon
dt = 1.5 / H  # time step
map_scale = 0.3


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

def plot_dyn(xs_batch, ys_batch, title = "", name="foo", xss=None):
    plt.figure()
    for xs, ys in zip(xs_batch, ys_batch):
        plt.plot(xs[:, 0], xs[:, 1], c="r", alpha=0.5)
        plt.scatter(
            [x[0] for x in ys],
            [x[1] for x in ys],
            label="observation",
            c=range(xs.shape[0]),
            cmap="Blues",
            marker="o",
            alpha=1.0,
        )
        plt.scatter(
            [x[0] for x in xs],
            [x[1] for x in xs],
            label="state",
            c=range(xs.shape[0]),
            cmap="Reds",
            marker="o",
            alpha=1.0,
        )
    if xss is not None:
        for i in range(xss.shape[0]):
            plt.plot(
                xss[i, :, 0],
                xss[i, :, 1],
                c="black",
                alpha=0.3,
                linewidth=1,
                linestyle="--",
            )
            plt.scatter(
                [x[0] for x in xss[i]],
                [x[1] for x in xss[i]],
                c=range(xss[i].shape[0]),
                cmap="Greys",
                marker="o",
                alpha=1.0,
            )
    # plot barrier with wide line
    plt.plot(
        [-2.0 * map_scale, 0.0, 0.0, -2.0 * map_scale],
        [-1.0 * map_scale, -1.0 * map_scale, 1.0 * map_scale, 1.0 * map_scale],
        "k",
        linewidth=10,
    )
    # set x, y limits
    plt.xlim(jnp.array([-4.0, 4.0])*map_scale)
    plt.ylim(jnp.array([-4.0, 4.0])*map_scale)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.title(title)
    # plt.legend()
    # save plot
    plt.savefig(f"../figure/{name}.png")
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
    return -0.5 * jnp.mean(((ys - xs) / sigma) ** 2, axis=[0, 1])


def get_logpc(xs):
    costs = jax.vmap(cost)(xs)
    # costs = costs.at[-1].set(jnp.clip(costs[-1], -(1.0/3.0)**2, 0.0) * 10.0)
    # costs = costs.at[0].set(jnp.clip(costs[0], -(1.0/3.0)**2, 0.0) * 10.0)
    costs = costs.at[-1].set(costs[-1] * 10.0)
    costs = costs.at[0].set(costs[0]*10.0)
    # costs = costs.at[0].set((jnp.linalg.norm(xs[0] - jnp.array([-1.0, 0.0])) ** 2)*100.0)
    logpc = -costs.mean()
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


# run MPPI
def mppi_traj(us, key):
    us_key, key = jax.random.split(key)
    us_batch = jax.random.normal(us_key, (Nx, H, 2)) * 1.0 + us
    xs_batch = jax.vmap(rollout_traj, in_axes=(None, 0))(
        jnp.array([-1.0, 0.0])*map_scale, us_batch
    )
    logpc = jax.vmap(get_logpc)(xs_batch)
    w_unnorm = jnp.exp((logpc - jnp.max(logpc))/0.1)
    w = w_unnorm / jnp.sum(w_unnorm, axis=0)
    us = jnp.sum(w[:, None, None] * us_batch, axis=0)
    return us, key, xs_batch[:8]


us = jnp.zeros((H, 1))
for i in range(1):
    us, key, xs_batch = mppi_traj(us, key)
    xs_mppi = rollout_traj(jnp.array([-1.0, 0.0])*map_scale, us)
    # plot_dyn(xs_mppi, xs_mppi, f"MPPI_{i}", xs_batch[:8])

# exit()

def denoise_traj(ys, us, sigma, key):
    # filter for new trajectory
    # all_moved = jnp.zeros((Nx), dtype=bool)
    # xs_batch = jnp.zeros((Nx, H+1, 2))
    # while not all_moved.all():
    us_key, key = jax.random.split(key)
    us_batch = us + jax.random.normal(us_key, (Nx, H, 2)) * sigma
    us_batch = jnp.clip(us_batch, -1.0, 1.0)
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
    #     -jnp.flip(us, axis=0) + jax.random.normal(us_inv_key, (Nx, H, 2)) * sigma * 1.0
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
    # logps = logpd*15.0 + logpc*0.5
    # logps = logpd*70.0 + logpc*5.0
    # logps = logpd*70.0 + logpc*5.0
    logps = logpd*70.0 + logpc*0.0

    w_unnorm = jnp.exp((logps - jnp.max(logps)))
    w = w_unnorm / jnp.sum(w_unnorm, axis=0)

    us_new = jnp.sum(w[:, None, None] * us_batch, axis=0)
    xs_new = jnp.sum(w[:, None, None] * xs_batch, axis=0)
    return xs_new, us_new, key, jnp.concat([xs_batch[:4], xs_batch[-4:]], axis=0)


us_key, key = jax.random.split(key)
us = jax.random.normal(us_key, (H, 2)) * 1.0
ys_key, key = jax.random.split(key)
xs_guess = jnp.linspace(-1.0, 1.0, H + 1)[:, None] * jnp.array([1.0, 0.0]) * map_scale
ys = jax.random.normal(ys_key, (H + 1, 2)) * 2.0 + xs_guess
ys = ys.at[0, :2].set(jnp.array([-1.0, 0.0]) * map_scale)
ys = ys.at[-1, :2].set(jnp.array([1.0, 0.0]) * map_scale)
# ys = ys.at[1, :2].set(jnp.array([-3.0, 1.3]))
# ys = ys.at[2, :2].set(jnp.array([0.4, 1.3]))
ys = ys.at[1, :2].set(jnp.array([-2.0, 2.0]) * map_scale)
ys = ys.at[2, :2].set(jnp.array([0.0, 2.0]) * map_scale)
ys_key, key = jax.random.split(key)

ys_batch = jax.random.normal(ys_key, (Ny, H + 1, 2)) * map_scale * 1.5 + xs_guess
ys_batch = ys_batch.at[:, 0, :2].set(jnp.array([-1.0, 0.0])*map_scale)
ys_batch = ys_batch.at[:, -1, :2].set(jnp.array([1.0, 0.0])*map_scale)

# set us to ys
# us = jnp.diff(ys, axis=0) / dt
us = jnp.zeros((H, 2))
us_batch = jnp.zeros((Ny, H, 2))

T = 50

def get_alpha_bar(t):
    # x = (t / T - 0.5) * 8.0
    x = (t/T)*6.0 - 3.0 # from 3.0 to -3.0
    return jax.nn.sigmoid(-x)

ts = jnp.arange(T + 1)
alpha_bars = get_alpha_bar(ts)
alphas = alpha_bars / jnp.roll(alpha_bars, 1)
# for (i, var) in enumerate(np.arange(0.1, 0.0, -var_step)):
key_batch = jax.random.split(key, Ny)
for i in range(T, 0, -1):
    alpha_bar_prev = alpha_bars[i - 1]
    alpha_bar = alpha_bars[i]
    alpha = alphas[i]

    var = 1 - alpha_bar
    var_cond = (1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)

    sigma = jnp.sqrt(var)
    print(f"t={i}, alpha={alpha:.2f}, alpha_bar={alpha_bar:.2f}, sigma={sigma:.2f}")
    # x|yi
    # xs, us, key, xs_batch = denoise_traj(ys, us, sigma, key)
    xs_batch, us_batch, key_batch, xs_sampled = jax.vmap(denoise_traj, in_axes=(0, 0, None, 0))(
        ys_batch, us_batch, sigma, key_batch
    )
    # key = key_batch[0]
    # us = us_batch.mean(axis=0)
    # us_batch = jnp.diff(xs_batch, axis=1) / dt
    # us_batch = jnp.clip(us_batch, -1.0, 1.0)
    # if i % 10 == 9:
    plot_dyn(xs_batch, ys_batch, f"t={i}, alpha={alpha:.2f}, alpha_bar={alpha_bar:.2f}, sigma={sigma:.2f}", f"denoise_{T-i}", xs_sampled[0])

    # if var <= var_step:
    #     sigma_ys = jnp.sqrt(var_step)
    # else:
    #     # sigma_ys = jnp.sqrt(1.0 / (1.0 / var_step + 1.0 / (var - var_step)))
    #     sigma_ys = jnp.sqrt(var_step)
    # yi-1|yi
    ys_key, key = jax.random.split(key)
    # ys = xs + (ys-xs)*(var-var_step)/(var) + jax.random.normal(ys_key, (H+1, 2)) * sigma_ys
    # xs = xs_ref
    # ys = (
    #     jnp.sqrt(alpha) * (1 - alpha_bar_prev) * ys
    #     + jnp.sqrt(alpha_bar_prev) * (1 - alpha) * xs
    # ) / (1 - alpha_bar) + jax.random.normal(ys_key, (H + 1, 2)) * jnp.sqrt(var_cond)
    ys_batch = (
        jnp.sqrt(alpha) * (1 - alpha_bar_prev) * ys_batch
        + jnp.sqrt(alpha_bar_prev) * (1 - alpha) * xs_batch
    ) / (1 - alpha_bar) + jax.random.normal(ys_key, (Ny, H + 1, 2)) * jnp.sqrt(var_cond)
    # print(jnp.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar), jnp.sqrt(alpha_bar_prev) * (1 - alpha) / (1 - alpha_bar), jnp.sqrt(var_cond))
    ys_batch = ys_batch.at[:, 0, :2].set(jnp.array([-1.0, 0.0])*map_scale)
    ys_batch = ys_batch.at[:, -1, :2].set(jnp.array([1.0, 0.0])*map_scale)