import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# parameters
dt = 0.4
key = jax.random.PRNGKey(1)
N = 2048  # sampled trajectory number
H = 50  # horizon

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
        jnp.array([-2.0, 1.0]), jnp.array([0.0, 1.0]), x, x_new
    )
    pass_wall2 = check_pass_wall(
        jnp.array([0.0, -1.0]), jnp.array([0.0, 1.0]), x, x_new
    )
    pass_wall3 = check_pass_wall(
        jnp.array([-2.0, -1.0]), jnp.array([0.0, -1.0]), x, x_new
    )
    pass_wall = pass_wall1 | pass_wall2 | pass_wall3

    x_new = jnp.where(pass_wall, x, x_new)

    return x_new


def cost(x):
    return jnp.sum((x - jnp.array([1.0, 0.0])) ** 2)


def plot_dyn(xs, ys, name="foo", xss = None):
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
    )
    plt.scatter(
        [x[0] for x in xs],
        [x[1] for x in xs],
        label="state",
        c=range(xs.shape[0]),
        cmap="Reds",
        marker="o",
        alpha=1.0
    )
    if xss is not None:
        for i in range(xss.shape[0]):
            plt.plot(xss[i, :, 0], xss[i, :, 1], c="black", alpha=0.2)
    # plot barrier with wide line
    plt.plot([-2.0, 0.0, 0.0, -2.0], [-1.0, -1.0, 1.0, 1.0], "k", linewidth=10)
    # set x, y limits
    plt.xlim([-3.0, 3.0])
    plt.ylim([-3.0, 3.0])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.legend()
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
    return xs


# us = jnp.array([[1.0, 0.0] for _ in range(10)])
# xs = rollout_traj(jnp.array([-1.0, 0.0]), us)
# plot_dyn(xs, us)


# get the likelihood of the trajectory
def get_logpd(ys, xs, sigma):
    return -0.5 * jnp.sum(((ys - xs) / sigma) ** 2, axis=[0, 1])


def get_logpc(xs):
    costs = jax.vmap(cost)(xs)
    costs = costs.at[-1].set(costs[-1] * 100.0)
    logpc = -costs.sum()
    return logpc


def get_logp(ys, xs, sigma, pc_weight=1.0):
    logpd = get_logpd(ys, xs, sigma)
    logpc = get_logpc(xs)
    return (logpc*pc_weight + logpd)


# run MPPI
us_key, key = jax.random.split(key)
us_batch = jax.random.normal(us_key, (N, H, 2)) * 1.0
xs_batch = jax.vmap(rollout_traj, in_axes=(None, 0))(jnp.array([-1.0, 0.0]), us_batch)
logpc = jax.vmap(get_logpc)(xs_batch)
w_unnorm = jnp.exp(logpc - jnp.max(logpc))
w = w_unnorm / jnp.sum(w_unnorm, axis=0)
us_mppi = jnp.sum(w[:, None, None] * us_batch, axis=0)
xs_mppi = rollout_traj(jnp.array([-1.0, 0.0]), us_mppi)
plot_dyn(xs_mppi, xs_mppi, "MPPI", xs_batch[:8])

# exit()

def denoise_traj(ys, us, sigma, key):
    # filter for new trajectory
    us_key, key = jax.random.split(key)
    us_batch = us + jax.random.normal(us_key, (N, H, 2)) * sigma * 2.0
    us_batch = jnp.clip(us_batch, -1.0, 1.0)
    xs_batch = jax.vmap(rollout_traj, in_axes=(None, 0))(jnp.array([-1.0, 0.0]), us_batch)
    # pc_weight change according to the sigma. smaller sigma (0.3->0.0) -> larger pc_weight (0.01->1.0)
    # pc_weight = jnp.clip(1.0 - sigma / 0.3, 0.0, 1.0)
    pc_weight = jnp.where(sigma < 0.3, 10.0, 0.01)
    logps = jax.vmap(get_logp, in_axes=(None, 0, None, None))(ys, xs_batch, sigma, pc_weight)
    w_unnorm = jnp.exp(logps - jnp.max(logps))
    w = w_unnorm / jnp.sum(w_unnorm, axis=0)
    us_new = jnp.sum(w[:, None, None] * us_batch, axis=0)
    xs_new = rollout_traj(jnp.array([-1.0, 0.0]), us_new)
    return xs_new, us_new, key, xs_batch[:8]

ys_key, key = jax.random.split(key)
us = jax.random.normal(ys_key, (H, 2)) * 1.0
ys = jax.random.normal(ys_key, (H, 2)) * 2.0
var_step = 0.01
for (i, var) in enumerate(np.arange(0.5, 0.1, -var_step)):
    sigma = jnp.sqrt(var)
    xs, us, key, xs_batch = denoise_traj(ys, us, sigma, key)
    # if i % 10 == 9:
    plot_dyn(xs, ys, f"denoise_{i}", xs_batch)

    if var <= var_step:
        sigma_ys = jnp.sqrt(var_step)
    else:
        sigma_ys = jnp.sqrt(1.0 / (1.0 / var_step + 1.0 / (var - var_step)))
    ys = xs + (ys-xs)*(var-var_step)/(var) + jax.random.normal(ys_key, (H, 2)) * sigma_ys
var_step = 0.001
for (i, var) in enumerate(np.arange(0.1, 0.0, -var_step)):
    sigma = jnp.sqrt(var)
    xs, us, key, xs_batch = denoise_traj(ys, us, sigma, key)
    # if i % 10 == 9:
    plot_dyn(xs, ys, f"denoise_{i+40}", xs_batch)

    if var <= var_step:
        sigma_ys = jnp.sqrt(var_step)
    else:
        sigma_ys = jnp.sqrt(1.0 / (1.0 / var_step + 1.0 / (var - var_step)))
    ys = xs + (ys-xs)*(var-var_step)/(var) + jax.random.normal(ys_key, (H, 2)) * sigma_ys