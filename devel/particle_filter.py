import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# parameters
dt = 0.2
key = jax.random.PRNGKey(0)
N = 10 # sampled trajectory number
H = 50 # horizon

# setup dynamics
def check_pass_wall(p1, p2, x1, x2):
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return (ccw(p1, x1, x2) != ccw(p2, x1, x2)) & (ccw(p1, p2, x1) != ccw(p1, p2, x2))

def f(x, u):
    x_new = x + dt * u

    # check if the particle passes the wall
    pass_wall1 = check_pass_wall(jnp.array([-2.0, 1.0]), jnp.array([0.0, 1.0]), x, x_new)
    pass_wall2 = check_pass_wall(jnp.array([0.0, -1.0]), jnp.array([0.0, 1.0]), x, x_new)
    pass_wall3 = check_pass_wall(jnp.array([-2.0, -1.0]), jnp.array([0.0, -1.0]), x, x_new)
    pass_wall = pass_wall1 | pass_wall2 | pass_wall3

    x_new = jnp.where(pass_wall, x, x_new)

    return x_new

def cost(x):
    return jnp.sum((x - jnp.array([1.0, 0.0]))**2)

def plot_dyn(xs, us, name = "foo"):
    plt.figure()
    plt.plot([x[0] for x in xs], [x[1] for x in xs], 'o')
    # plot barrier with wide line
    plt.plot([-2.0, 0.0, 0.0, -2.0], [-1.0, -1.0, 1.0, 1.0], 'k', linewidth=10)
    # set x, y limits
    plt.xlim([-3.0, 3.0])
    plt.ylim([-3.0, 3.0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    # save plot
    plt.savefig(f'../figure/{name}.png')

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
def logp(ys, xs, sigma):
    # gaussian likelihood
    logpd = -0.5 * jnp.sum(((ys - xs)/sigma)**2, axis=[0, 1])
    # cost likelihood
    costs = jax.vmap(cost)(xs)
    costs = costs.at[-1].set(costs[-1] * 1000.0)
    logpc = -costs.sum()
    return logpc + logpd

# generate states
y_key, key = jax.random.split(key)
ys = jax.random.normal(y_key, (H, 2)) * 2.0
# sample trajectories
us_key, key = jax.random.split(key)
us_batch = jax.random.uniform(us_key, (N, H, 2)) * 2.0 - 1.0
xs_batch = jax.vmap(rollout_traj, in_axes=(None, 0))(jnp.array([-1.0, 0.0]), us_batch)
# get likelihood
sigma = 2.0
logps = jax.vmap(logp, in_axes=(None, 0, None))(ys, xs_batch, sigma)
logps = logps - jnp.max(logps)
w_unnorm = jnp.exp(logps)
w = w_unnorm / jnp.sum(w_unnorm, axis=0)
# get new weighted us
us_new = jnp.sum(w[:, None, None] * us_batch, axis=0)
# rollout the new trajectory
xs_new = rollout_traj(jnp.array([-1.0, 0.0]), us_new)
plot_dyn(xs_new, us_new, "01")

# add noise back
sigma = 0.6
ys = xs_new + jax.random.normal(y_key, (H, 2)) * sigma
# sample trajectories
us_batch = us_new + jax.random.normal(us_key, (N, H, 2)) * 1.0
xs_batch = jax.vmap(rollout_traj, in_axes=(None, 0))(jnp.array([-1.0, 0.0]), us_batch)
# get likelihood
logps = jax.vmap(logp, in_axes=(None, 0, None))(ys, xs_batch, sigma)
logps = logps - jnp.max(logps)
w_unnorm = jnp.exp(logps)
w = w_unnorm / jnp.sum(w_unnorm, axis=0)
# get new weighted us
us_new = jnp.sum(w[:, None, None] * us_batch, axis=0)
# rollout the new trajectory
xs_new = rollout_traj(jnp.array([-1.0, 0.0]), us_new)
plot_dyn(xs_new, us_new, "02")