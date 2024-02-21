import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

rng = jax.random.PRNGKey(0)

u_max = 1.5
x0 = jnp.array([-1.0, 0.0])
xg = jnp.array([1.0, 0.0])
x_obs = jnp.array([0.0, 0.0])
a_obs = 0.8

# create initial u and x as random
u0 = jax.random.uniform(rng, (2,), minval=-u_max, maxval=u_max)
u1 = jax.random.uniform(rng, (2,), minval=-u_max, maxval=u_max)
x1 = jax.random.uniform(rng, (2,), minval=-1.0, maxval=1.0)
x2 = jax.random.uniform(rng, (2,), minval=-1.0, maxval=1.0)

# plot initial trajectory
fig, ax = plt.subplots()
def plot_traj(ax, x0, x1, x2, xg, x_obs, a_obs):
    ax.plot([x0[0], x1[0], x2[0], xg[0]], [x0[1], x1[1], x2[1], xg[1]], 'bo-')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.add_patch(Rectangle((x_obs[0]-a_obs, x_obs[1]-a_obs), 2*a_obs, 2*a_obs, color='r'))
    ax.set_aspect('equal', 'box')
plot_traj(ax, x0, x1, x2, xg, x_obs, a_obs)

# rollout the trajectory with real dynamics
def dynamics(x0, u0, u1, x_obs, r_obs):
    # simulate the dynamics
    # if hit the obstacle, clip it out
    def get_next_x(x0, u0, x_obs, r_obs):
        x1 = x0 + u0
        x1 = x1.at[0].set(jnp.where(x0[0]<x_obs[0], jnp.minimum(x1[0], x_obs[0]-r_obs), jnp.maximum(x1[0], x_obs[0]+r_obs)))
        x1 = x1.at[1].set(jnp.where(x0[1]<x_obs[1], jnp.minimum(x1[1], x_obs[1]-r_obs), jnp.maximum(x1[1], x_obs[1]+r_obs)))
        return x1
    x1 = get_next_x(x0, u0, x_obs, r_obs)
    x2 = get_next_x(x1, u1, x_obs, r_obs)
    return x1, x2

# get the cost
def cost(x1, x2, xg):
    return jnp.sum((x1-xg)**2) + jnp.sum((x2-xg)**2)

# update the trajectory
def update_traj(x0, x1, x2, xg, x_obs, a_obs, u0, u1, lr=0.1):
    x1_dyn, x2_dyn = dynamics(x0, u0, u1, x_obs, a_obs)
    cost0 = cost(x1_dyn, x2_dyn, xg)
    grad0 = jax.grad(cost, (0, 1))(x1, x2, xg)
    

# plot the trajectory
plot_traj(ax, x0, x1_dyn, x2_dyn, xg, x_obs, a_obs)
plt.show()