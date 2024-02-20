import numpy as np
# from jax import numpy as jnp
from matplotlib import pyplot as plt

def main():
    # load data npz
    data = np.load('diffuse_traj.npz')
    x = data['x_traj_save'] # shape (n_diffuse_step, n_traj, n_step, n_dim)
    x = x[np.newaxis, :, :, :]  # shape (1, n_diffuse_step, n_traj, n_step, n_dim)
    print(x[0,0,0,:2])
    # generate color map from white to red
    for d_step in range(0, x.shape[0]):
        plt.figure()
        plt.gca().set_aspect('equal', adjustable='box')
        # plot 2D traj
        plot_traj = 0
        xs = x[d_step, plot_traj, :, 0]
        ys = x[d_step, plot_traj, :, 1]
        thetas = x[d_step, plot_traj, :, 2]
        # plot arrow at each step, with white to red color map
        plt.quiver(xs, ys, -np.sin(thetas), np.cos(thetas), range(len(xs)), cmap='Reds')
        plt.grid()
        # plt.xlim([0, 5])
        # plt.ylim([-2, 2])
        plt.savefig(f'figure/traj_{d_step}.png')



if __name__ == "__main__":
    main()