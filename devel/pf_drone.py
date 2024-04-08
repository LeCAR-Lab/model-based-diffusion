import jax
from jax import numpy as jnp
from jax import scipy as jsc
from jax.scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

# parameters
key = jax.random.PRNGKey(1)
N = 1024  # sampled trajectory number
H = 10  # horizon
dt = 0.1
n_plot_samples = 8
# dynamics parameters
J = jnp.array([[1.0/20.0, 0.0, 0.0], [0.0, 1.0/20.0, 0.0], [0.0, 0.0, 1.0/4.0]])
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
xf = jnp.array([
    0.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0,
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
    Rq = R.from_quat(q)

    Re = Rd.inv() * Rq
    # convert to axis-angle
    Rotvec_e = Re.as_rotvec()
    # qe = Re.as_quat()
    # Mat_e = Re.as_matrix()

    # c1 = jnp.sum(vee(Mat_e - Mat_e.T)**2 / (1+jnp.trace(Mat_e))) # NOTE: add small value to avoid nan
    # c1 = jnp.sum((q - qd)**2)*0.5
    c1 = jnp.sum((Rotvec_e/jnp.pi)**2)
    c2 = jnp.sum(w**2)

    cost = (c1 + 0.01*c2)

    return cost


def plot_dyn(xs, ys, name="foo", xss=None, costs = None, us = None):
    # create 5 subplots
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    qxs = xs[:, 3:7]
    qys = ys[:, 3:7]
    colors = ["r", "g", "b", "y"]
    labels = ["x", "y", "z", "w"]
    if us is not None:
        for i in range(3):
            ax = axs[i]
            ax.plot(us[:, i+1], c = "k", label = "u")
    for i in range(4):
        ax = axs[i]
        # set title
        ax.set_title(f"q_{labels[i]}")
        # plot with red
        ax.plot(qxs[:, i], c = colors[i], label = "x")
        ax.plot(qys[:, i], c = colors[i], linestyle="--", label = "y") 
        ax.axhline(qd[i], c = 'k', linestyle="--")
        ax.legend()
        ax.set_ylim([-1.0, 1.0])
    if costs is not None:
        axs[4].set_title("cost")
        axs[4].plot(costs)
    if xss is not None:
        for i in range(xss.shape[0]):
            qxs = xss[i, :, 3:7]
            for j in range(4):
                ax = axs[j]
                ax.plot(qxs[:, j], c = colors[j], alpha=0.1)
    # save plot
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
    qxs = xs[:, 3:7]
    wxs = xs[:, 10:]
    qys = ys[:, 3:7]
    wys = ys[:, 10:]
    
    Rqxs = jax.vmap(R.from_quat)(qxs)
    Rqys = jax.vmap(R.from_quat)(qys)
    Rerrs = jax.vmap(lambda Rqy, Rqx: Rqy.inv() * Rqx)(Rqys, Rqxs)
    Rotvec_errs = jax.vmap(lambda Rerr: Rerr.as_rotvec())(Rerrs)
    # d1 = jnp.sum(((qxs - qys))**2, axis=1).mean()
    d1 = jnp.sum((Rotvec_errs/jnp.pi)**2, axis=1).mean()
    d2 = jnp.sum(((wxs - wys)/7.0)**2, axis=1).mean()
    return - d1*5.0 - d2*0.0

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

def denoise_traj(ys, us, sigma, key):
    # filter for new trajectory
    us_key, key = jax.random.split(key)
    us_batch = jnp.clip(us + jax.random.normal(us_key, (N, H, nu)) * sigma * 1.0, -1.0, 1.0)
    xs_batch = jax.vmap(rollout_traj, in_axes=(None, 0))(x0, us_batch)

    logpd = jax.vmap(get_logpd, in_axes=(None, 0, None))(ys, xs_batch, sigma)
    logpc = jax.vmap(get_logpc)(xs_batch)
    logps = logpd*8.0 + logpc*8.0

    w_unnorm = jnp.exp((logps - jnp.max(logps)))
    w = w_unnorm / jnp.sum(w_unnorm, axis=0)

    # print(f"logpd: {logpd.mean():.2f} \pm {logpd.std():.2f} logpc: {logpc.mean():.2f} \pm {logpc.std():.2f}")

    us_new = jnp.sum(w[:, None, None] * us_batch, axis=0)
    xs_new = jnp.sum(w[:, None, None] * xs_batch, axis=0)
    return xs_new, us_new, key, xs_batch[:8]


# us_key, key = jax.random.split(key)
# us = jax.random.normal(us_key, (H, nu)) * 1.0
ys_key, key = jax.random.split(key)
rpys = jnp.zeros((H+1, 3))
# H2 = (H+1)//2
# rpys = rpys.at[:H2, 0].set(jnp.linspace(0.0, jnp.pi, H2))
# rpys = rpys.at[H2:, 0].set(jnp.linspace(jnp.pi, jnp.pi, H+1-H2))
# rpys = rpys.at[:H2, 1].set(jnp.linspace(0.0, 0.0, H2))
# rpys = rpys.at[H2:, 1].set(jnp.linspace(0.0, -jnp.pi, H+1-H2))
# rpys = rpys.at[:, 0].set(jnp.linspace(0.0, jnp.pi, H+1))
# rpys = rpys.at[:, 1].set(jnp.linspace(0.0, -jnp.pi, H+1))
# qs = jax.vmap(lambda rpy: R.from_euler("xyz", rpy).as_quat())(rpys)
# qdots = jnp.diff(qs, axis=0, prepend=qs[:1]) / dt
# q_dot = 0.5 * L(q) @ Hmat @ w
# Lquats = jax.vmap(L)(qs)
# ws = 2.0 * jax.vmap(lambda Lquats, qdot: jnp.linalg.inv(Lquats) @ qdot)(Lquats, qdots)[:, :3]
def sample_quat(key):
    u = jax.random.uniform(key, (3,))
    q = jnp.array([
        jnp.sqrt(1 - u[0]) * jnp.sin(2 * jnp.pi * u[1]),
        jnp.sqrt(1 - u[0]) * jnp.cos(2 * jnp.pi * u[1]),
        jnp.sqrt(u[0]) * jnp.sin(2 * jnp.pi * u[2]),
        jnp.sqrt(u[0]) * jnp.cos(2 * jnp.pi * u[2])
    ])
    return q
ys = jnp.zeros((H+1, 13))
quat_key, key = jax.random.split(key)
ys = ys.at[:, 3:7].set(jax.vmap(sample_quat)(jax.random.split(quat_key, H+1)))
ys = ys.at[0].set(x0)
ys = ys.at[-1].set(xf)
# ys = ys.at[:, 3:7].set(qs)
# ys = ys.at[:, 10:].set(ws)
# set us to ys
# wdots = jnp.diff(ws, axis=0) / dt
# torques = jax.vmap(lambda wdot, w: J @ wdot - jnp.cross(J@w, w))(wdots, ws[:-1])
# torques = jnp.clip(torques, -1.0, 1.0)
us = jnp.zeros((H, nu))
# us = us.at[:, 1:].set(torques)
# us = us.at[:, 1].set(1.0)
# us = us.at[:, 2].set(1.0)
xs = rollout_traj(x0, us)
plot_dyn(xs, ys, "init", costs=jax.vmap(cost)(ys), us=us)

T = 500

def get_alpha_bar(t):
    # x = (t / T - 0.5) * 8.0
    x = (t/T)*7.0 - 5.0 # from 2.0 to -5.0
    return jax.nn.sigmoid(-x)


ts = jnp.arange(T + 1)
alpha_bars = get_alpha_bar(ts)
alphas = alpha_bars / jnp.roll(alpha_bars, 1)
# for (i, var) in enumerate(np.arange(0.1, 0.0, -var_step)):
denoise_traj_jit = jax.jit(denoise_traj)
for i in range(T, 0, -1):
    alpha_bar_prev = alpha_bars[i - 1]
    alpha_bar = alpha_bars[i]
    alpha = alphas[i]

    var = 1 - alpha_bar
    var_cond = (1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)

    sigma = jnp.sqrt(var)
    # x|yi
    xs, us, key, xs_batch = denoise_traj_jit(ys, us, sigma, key)
    cs = jax.vmap(cost)(xs)
    if i % 100 == 99:
        plot_dyn(xs, ys, "denoise", xs_batch, cs, us)

    # yi-1|yi
    ys_key, key = jax.random.split(key)
    ys = (
        jnp.sqrt(alpha) * (1 - alpha_bar_prev) * ys
        + jnp.sqrt(alpha_bar_prev) * (1 - alpha) * xs
    ) / (1 - alpha_bar) + jax.random.normal(ys_key, (H + 1, 13)) * jnp.sqrt(var_cond)
    # print(jnp.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar), jnp.sqrt(alpha_bar_prev) * (1 - alpha) / (1 - alpha_bar), jnp.sqrt(var_cond))
    ys = ys.at[0].set(x0)
    ys = ys.at[-1].set(xf)
    ys = ys.at[:, 3:7].set(jax.vmap(lambda q: q / jnp.linalg.norm(q))(ys[:, 3:7]))
# save xs
jnp.save(f"../figure/xs.npy", xs)