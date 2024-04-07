import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# parameters
key = jax.random.PRNGKey(2)
N = 1024  # sampled trajectory number
H = 5  # horizon
map_scale = 0.5
# dt = 5.1 / H * map_scale
dt = 10.0 / H * map_scale


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
def rollout_traj(x0, us, sigma_dyn, key):
    # rollout with jax.lax.scan
    def step(carry, u):
        x, key = carry
        key_dyn, key = jax.random.split(key)
        wd = jax.random.normal(key_dyn, (2,)) * sigma_dyn
        x_new = f(x, u) + wd
        return (x_new, key), (x_new, wd)

    _, (xs, wds) = jax.lax.scan(step, (x0, key), us)
    xs = jnp.concatenate([x0[None, :], xs], axis=0)
    wds = jnp.concatenate([jnp.zeros((1, 2)), wds], axis=0)
    return xs, wds


# us = jnp.array([[1.0, 0.0] for _ in range(10)])
# xs = rollout_traj(jnp.array([-1.0, 0.0]), us)
# plot_dyn(xs, us)


# get the likelihood of the trajectory
def get_mppi_logp(xs, wns, sigma_dyn):
    final_cost = jnp.clip(jnp.sum((xs[-1]/map_scale - jnp.array([1.0, 0.0])) ** 2), 0.0, 1.0)*3.0
    dynamic_cost = jnp.mean((wns/sigma_dyn)**2)*0.0
    return -final_cost - dynamic_cost

# run MPPI
def mppi_traj(us, key, sigma_dyn):
    us_key, key = jax.random.split(key)
    us_batch = jax.random.normal(us_key, (N, H, 2)) * 1.0 + us
    us_batch = jnp.clip(us_batch, -1.0, 1.0)

    key_traj, key = jax.random.split(key)
    key_rollout = jax.random.split(key_traj, N)
    xs_batch, ws_batch = jax.vmap(rollout_traj, in_axes=(None, 0, None, 0))(
        jnp.array([-1.0, 0.0])*map_scale, us_batch, sigma_dyn, key_rollout
    )

    logpc = jax.vmap(get_mppi_logp, in_axes=(0,0,None))(xs_batch, ws_batch, sigma_dyn)*8.0
    logw = logpc - jnp.max(logpc)
    jax.debug.print("logw = {x} \pm {y}", x=jnp.mean(logw), y=jnp.std(logw))
    w_unnorm = jnp.exp(logw)
    w = w_unnorm / jnp.sum(w_unnorm, axis=0)
    us = jnp.sum(w[:, None, None] * us_batch, axis=0)
    return us, key, xs_batch[:8]


us = jnp.zeros((H, 2))
sigma_dyns = jnp.linspace(0.01, 0.01, 30)
for (i, sigma_dyn) in enumerate(sigma_dyns):
    us, key, xs_batch = mppi_traj(us, key, sigma_dyn)
    xs_mppi = rollout_traj(jnp.array([-1.0, 0.0])*map_scale, us, sigma_dyn, key)[0]
    plot_dyn(xs_mppi, xs_mppi, f"MPPI_{i}", xs_batch[:8])