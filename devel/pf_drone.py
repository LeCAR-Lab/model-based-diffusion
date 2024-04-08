import jax
from jax import numpy as jnp
from jax import scipy as jsc
from jax.scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

# parameters
key = jax.random.PRNGKey(2)
N = 1024  # sampled trajectory number
H = 20  # horizon
dt = 0.05
n_plot_samples = 8
# dynamics parameters
J = jnp.array([[1.0/16.0, 0.0, 0.0], [0.0, 1.0/16.0, 0.0], [0.0, 0.0, 1.0/4.0]])
J_inv = jnp.linalg.inv(J)
# desired rotation
Rd = R.from_euler("xyz", [0.0, 0.0, jnp.pi])
qd = Rd.as_quat()
# initial state
x0 = jnp.array([
    0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0
])
# dims
nu = 4

# setup dynamics
# quaternion (x, y, z, w)
Hmat = jnp.vstack((jnp.eye(3), jnp.zeros((1, 3))))

def vee(R):
    return jnp.array([R[2, 1], R[0, 2], R[1, 0]])

def hat(v: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def L(q: jnp.ndarray) -> jnp.ndarray:
    s = q[3]
    v = q[:3]
    right = jnp.hstack((v, s)).reshape(-1, 1)
    left_up = s * jnp.eye(3) + hat(v)
    left_down = -v
    left = jnp.vstack((left_up, left_down))
    return jnp.hstack((left, right))

def f(x, u):
    # state
    r = x[:3]
    q = x[3:7]
    v = x[7:10]
    w = x[10:]
    # control
    thrust = u[0]
    torque = u[1:]
    # dot state
    r_dot = jnp.zeros(3)
    q_dot = 0.5 * L(q) @ Hmat @ w
    v_dot = jnp.zeros(3)
    w_dot = J_inv @ (jnp.cross(J@w, w) + torque)
    return jnp.concatenate([r_dot, q_dot, v_dot, w_dot])

def rk4(x, u):
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    x_new = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # normalize quaternion
    q = x_new[3:7]
    q = q / jnp.linalg.norm(q)
    x_new = x_new.at[3:7].set(q)
    return x_new

def cost(x):
    r = x[:3]
    q = x[3:7]
    v = x[7:10]
    w = x[10:]
    # Rq = R.from_quat(q)

    # Re = Rd.inv() * Rq
    # qe = Re.as_quat()
    # Mat_e = Re.as_matrix()

    # c1 = jnp.sum(vee(Mat_e - Mat_e.T)**2 / (1+jnp.trace(Mat_e))) # NOTE: add small value to avoid nan
    c1 = jnp.sum((q - qd)**2)*0.5
    c2 = jnp.sum(w**2)

    cost = (c1 + 0.01*c2)

    return cost


def plot_dyn(xs, ys, name="foo", xss=None, costs = None):
    # create 5 subplots
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    qxs = xs[:, 3:7]
    qys = ys[:, 3:7]
    colors = ["r", "g", "b", "y"]
    labels = ["x", "y", "z", "w"]
    for i in range(4):
        ax = axs[i]
        # set title
        ax.set_title(f"q_{labels[i]}")
        # plot with red
        ax.plot(qxs[:, i], c = colors[i])
        ax.plot(qys[:, i], c = colors[i], linestyle="--")
        ax.axhline(qd[i], c = 'k', linestyle="--")
    if costs is not None:
        axs[4].plot(costs, label="cost")
    if xss is not None:
        for i in range(xss.shape[0]):
            qxs = xss[i, :, 3:7]
            for j in range(4):
                ax = axs[j]
                ax.plot(qxs[:, j], c = colors[j], alpha=0.1)
    # save plot
    plt.legend()
    plt.savefig(f"../figure/{name}.png")
    plt.close()


# simulate the dynamics
def rollout_traj(x0, us):
    # rollout with jax.lax.scan
    def step(x, u):
        x_new = rk4(x, u)
        return x_new, x_new

    _, xs = jax.lax.scan(step, x0, us)
    return jnp.concatenate([x0[None, :], xs], axis=0)


# u0 = jnp.array([0.0, 0.0, 0.0, 1.0])
# us = jnp.repeat(u0[None, :], H, axis=0)
# xs = rollout_traj(x0, us)
# qs = xs[:, 3:7]
# cs = jax.vmap(cost)(xs)
# plot_dyn(xs, xs, "foo", costs=cs)
# exit()

# get the likelihood of the trajectory
def get_logpd(ys, xs, sigma):
    return -0.5 * jnp.mean((((ys - xs) / sigma)[1:] ** 2).sum(axis=1), axis=0)

def get_logpc(xs):
    logpc = - jax.vmap(cost)(xs).mean()
    return logpc


# run MPPI
def mppi_traj(us, key):
    us_key, key = jax.random.split(key)
    us_batch = jax.random.normal(us_key, (N, H, nu)) * 1.0 + us
    us_batch = jnp.clip(us_batch, -1.0, 1.0)
    xs_batch = jax.vmap(rollout_traj, in_axes=(None, 0))(
        x0, us_batch
    )
    logpc = jax.vmap(get_logpc)(xs_batch)*8.0
    w_unnorm = jnp.exp((logpc - jnp.max(logpc)))
    w = w_unnorm / jnp.sum(w_unnorm, axis=0)
    us = jnp.sum(w[:, None, None] * us_batch, axis=0)
    return us, key, xs_batch[:n_plot_samples]


us = jnp.zeros((H, nu))
for i in range(1):
    us, key, xs_batch = mppi_traj(us, key)
    xs_mppi = rollout_traj(x0, us)
    cs = jax.vmap(cost)(xs_mppi)
    plot_dyn(xs_mppi, xs_mppi, f"MPPI_{i}", xs_batch, cs)

exit()

plt.figure()
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
    plt.hist(w, bins=20)
    plt.savefig("../figure/w.png")
    plt.cla()

    print(f"logpd: {logpd.mean():.2f} \pm {logpd.std():.2f} logpc: {logpc.mean():.2f} \pm {logpc.std():.2f}")

    # plot w with histogram
    # plt.hist(w, bins=20)
    # plt.show()
    # plt.cla()

    us_new = jnp.sum(w[:, None, None] * us_batch, axis=0)
    xs_new = jnp.sum(w[:, None, None] * xs_batch, axis=0)
    return xs_new, us_new, key, xs_batch[:8]


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

T = 300

def get_alpha_bar(t):
    # x = (t / T - 0.5) * 8.0
    x = (t/T)*6.0 - 3.0 # from 2.0 to -5.0
    return jax.nn.sigmoid(-x)


ts = jnp.arange(T + 1)
alpha_bars = get_alpha_bar(ts)
alphas = alpha_bars / jnp.roll(alpha_bars, 1)
# for (i, var) in enumerate(np.arange(0.1, 0.0, -var_step)):
for i in range(T, 0, -1):
    alpha_bar_prev = alpha_bars[i - 1]
    alpha_bar = alpha_bars[i]
    alpha = alphas[i]

    var = 1 - alpha_bar
    var_cond = (1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)

    sigma = jnp.sqrt(var)
    # x|yi
    xs, us, key, xs_batch = denoise_traj(ys, us, sigma, key)
    # us = jnp.diff(ys, axis=0) / dt
    # us = jnp.clip(us, -1.0, 1.0)
    # if i % 10 == 9:
    plot_dyn(xs, ys, f"denoise_{T-i}", xs_batch)

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
    print(jnp.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar), jnp.sqrt(alpha_bar_prev) * (1 - alpha) / (1 - alpha_bar), jnp.sqrt(var_cond))
    ys = ys.at[0, :2].set(jnp.array([-1.0, 0.0])*map_scale)
    ys = ys.at[-1, :2].set(jnp.array([1.0, 0.0])*map_scale)
    # key_yf, key = jax.random.split(key)
    # ys = ys.at[-1, :2].set((jnp.array([1.0, 0.0])+jax.random.normal(key_yf, (2,))*sigma)*map_scale)