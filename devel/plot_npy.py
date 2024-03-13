from matplotlib import pyplot as plt
from jax import numpy as jnp
import argparse

parser = argparse.ArgumentParser()

x_trajs = jnp.load("figure/x_traj.npy")
x_traj_reals = jnp.load("figure/x_traj_real.npy")

# input start and end point as int
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=x_trajs.shape[0])
args = parser.parse_args()


obstacle = 'umaze'
task = 'drone'

horizon = x_trajs.shape[2]

fig, ax = plt.subplots(1, 1)
# for info in save_infos:
for i in range(args.start, args.end):
    x_traj = x_trajs[i]
    x_traj_real = x_traj_reals[i]
    if obstacle == 'square':
        rect = plt.Rectangle((-0.2, -0.2), 0.4, 0.4, color="black", fill=False)
        ax.add_artist(rect)
    elif obstacle == 'wall':
        rect = plt.Rectangle((-0.2, -0.5), 0.4, 1.0, color="black", fill=False)
        ax.add_artist(rect)
    elif obstacle == 'umaze':
        rect1 = plt.Rectangle((-1.0-0.5, 1.0-0.5), 2.0, 1.0, color="black", fill=True)
        rect2 = plt.Rectangle((-0.5, -1.0-0.5), 1.0, 3.0, color="black", fill=True)
        rect3 = plt.Rectangle((-1.0-0.5, -1.0-0.5), 2.0, 1.0, color="black", fill=True)
        ax.add_artist(rect1)
        ax.add_artist(rect2)
        ax.add_artist(rect3)
    elif obstacle == 'sphere':
        circle = plt.Circle((0, 0), 0.5, color="black", fill=False)
        ax.add_artist(circle)
    for j in range(x_traj.shape[0]):
        if task == 'point':
            ax.scatter(
                x_traj[j, :, 0],
                x_traj[j, :, 1],
                c=range(horizon),
                cmap="Reds",
                marker="o",
                alpha=1.0,
            )
        elif task == 'drone':
            ax.quiver(
                x_traj[j, :, 0],
                x_traj[j, :, 1],
                jnp.cos(x_traj[j, :, 2]),
                jnp.sin(x_traj[j, :, 2]),
                range(horizon),
                cmap="Reds",
                alpha=1.0,
            )
        ax.plot(
            x_traj[j, :, 0],
            x_traj[j, :, 1],
            "r",
            alpha=0.2,
        )
        ax.plot(
            x_traj_real[j, :, 0],
            x_traj_real[j, :, 1],
            "b--",
        )
    ax.grid()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect("equal", adjustable="box")
    # plot star at [1, 0]
    ax.plot(1.0, 0.0, "r*", markersize=16)
    # set title
    ax.set_title("Trajectory")
    # save figure to file
    plt.savefig(f"figure/{i}.png")
    
    ax.clear()

