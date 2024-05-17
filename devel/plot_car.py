from matplotlib import pyplot as plt
from jax import numpy as jnp
import jax
jax.config.update("jax_platform_name", "cpu")

r_obs = 0.3
obs_center = jnp.array(
    [
        [-r_obs * 3, r_obs * 2],
        [-r_obs * 2, r_obs * 2],
        [-r_obs * 1, r_obs * 2],
        [0.0, r_obs * 2],
        [0.0, r_obs * 1],
        [0.0, 0.0],
        [0.0, -r_obs * 1],
        [-r_obs * 3, -r_obs * 2],
        [-r_obs * 2, -r_obs * 2],
        [-r_obs * 1, -r_obs * 2],
        [0.0, -r_obs * 2],
    ]
)
obs_radius = r_obs  # Radius of the obstacle
H = 50

def render(ax, xs: jnp.ndarray, with_label=True):
    # obstacles
    for i in range(obs_center.shape[0]):
        circle = plt.Circle(
            obs_center[i, :], obs_radius, color="k", fill=True, alpha=0.5
        )
        ax.add_artist(circle)
    ax.scatter(xs[:, 0], xs[:, 1], c=range(H + 1), cmap="Reds")
    if with_label:
        ax.plot(xs[:, 0], xs[:, 1], "r-", label="Car path")
    else:
        ax.plot(xs[:, 0], xs[:, 1], "r-")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.grid(True)

xs0 = jnp.load("../results/car2d/xs0.npy")
xref0 = jnp.load("../results/car2d/xref0.npy")
xs1 = jnp.load("../results/car2d/xs1.npy")
xref1 = jnp.load("../results/car2d/xref1.npy")

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(xref0[:, 0], xref0[:, 1], "g--", label="RRT path")
ax.plot(xref1[:, 0], xref1[:, 1], "g--")
render(ax, xs0)
render(ax, xs1, with_label=False)
ax.legend()
plt.savefig("../results/car2d/rollout.png")