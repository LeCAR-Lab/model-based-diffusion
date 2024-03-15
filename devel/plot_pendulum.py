from matplotlib import pyplot as plt
from jax import numpy as jnp
import argparse

# jax set to cpu
import jax
jax.config.update("jax_platform_name", "cpu")

parser = argparse.ArgumentParser()

xs_true = jnp.load("../figure/states.npy")
xs_ekf = jnp.load("../figure/ekf_filtered_means.npy")
xs_eks = jnp.load("../figure/ekf_smoothed_means.npy")

# input start and end point as int
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=xs_true.shape[0])
args = parser.parse_args()

horizon = xs_true.shape[1]

fig, axs = plt.subplots(1, 3)
# for info in save_infos:
for i in range(args.start, args.end):
    for j, ax in enumerate(axs):
        xs = [xs_true, xs_ekf, xs_eks][j]
        theta = xs[i, 0]
        x = jnp.sin(theta) * 1.5
        y = -jnp.cos(theta) * 1.5
        ax.plot([0, x], [0, y], "r")
        ax.scatter(x, y, c="b", marker="o")
        ax.grid()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_aspect("equal", adjustable="box")
        # set title
        ax.set_title(["True", "EKF", "EKS"][j])
    # save figure to file
    plt.savefig(f"../figure/{i}.png")
    # clear axes plots
    for ax in axs:
        ax.clear()