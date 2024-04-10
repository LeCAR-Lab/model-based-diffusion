from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax
from jax import lax
from flax import struct
import chex
from tqdm import trange

import matplotlib
import matplotlib.pyplot as plt

from importlib import reload

from trajax import integrators
from trajax.experimental.sqp import shootsqp, util

reload(shootsqp)
reload(util)

# global parameters
nx, nu = (4, 2)
dt, T = (0.1, 30)
Ndiff = 32
Niter = 1000
x0 = jnp.array([-1.0, 0.0, 0.0, 0.0])
xT = jnp.array([1.0, 0.0, 0.0, 0.0])


def generate_noise_schedule(init, final, steps):
    scale = 8.0
    noise_var_schedule = jnp.exp(jnp.linspace(scale, 0.0, steps)) / jnp.exp(scale)
    noise_var_schedule = noise_var_schedule * (init - final) + final
    return noise_var_schedule


noise_var_schedule = generate_noise_schedule(1.0, 1e-4, Ndiff)
eps_schedule = jnp.linspace(30.0, 0.01, Niter) * 1e-6

obs = [
    (jnp.array([0.0, 0.0]), 0.5),
    # (jnp.array([1, 2.5]), 0.5),
    # (jnp.array([2.5, 2.5]), 0.5),
]

key = jax.random.PRNGKey(0)


def render_scene():
    # Setup obstacle environment for state constraint
    # world_range = (jnp.array([-0.5, -0.5]), jnp.array([3.5, 3.5]))
    world_range = (jnp.array([-2.0, -2.0]), jnp.array([2.0, 2.0]))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    plt.grid(False)

    for ob in obs:
        ax.add_patch(plt.Circle(ob[0], ob[1], color="k", alpha=0.3))
    ax.set_xlim([world_range[0][0], world_range[1][0]])
    ax.set_ylim([world_range[0][1], world_range[1][1]])

    ax.set_aspect("equal")
    return fig, ax


# Setup discrete-time dynamics
def car_ode(x, u, t):
    # x: [x, y, theta, v]
    del t
    return jnp.array([x[3] * jnp.sin(x[2]), x[3] * jnp.cos(x[2]), x[3] * u[0], u[1]])


dynamics = integrators.rk4(car_ode, dt)

# Cost function.
R = jnp.diag(jnp.array([0.2, 0.1]))
Q_T = jnp.diag(jnp.array([50.0, 50.0, 50.0, 10.0]))

# Indices of state corresponding to S1 sphere constraints
s1_indices = (2,)
state_wrap = util.get_s1_wrapper(s1_indices)


@jax.jit
def cost(x, u, t):
    stage_cost = dt * jnp.vdot(u, R @ u)
    delta = state_wrap(x - xT)
    term_cost = jnp.vdot(delta, Q_T @ delta)
    return jnp.where(t == T, term_cost, stage_cost)


# Control box bounds
control_bounds = (jnp.array([-jnp.pi / 3.0, -6.0]), jnp.array([jnp.pi / 3.0, 6.0]))


def unnorm_control(yu):
    return (
        yu * (control_bounds[1] - control_bounds[0]) / 2.0
        + (control_bounds[1] + control_bounds[0]) / 2.0
    )


def unnorm_state(yx):
    return yx


# Obstacle avoidance constraint function
def obs_constraint(pos):
    def avoid_obs(pos_c, ob):
        delta_body = pos_c - ob[0]
        delta_dist_sq = jnp.vdot(delta_body, delta_body) - (ob[1] ** 2)
        return delta_dist_sq

    return jnp.array([avoid_obs(pos, ob) for ob in obs])


# State constraint function
@jax.jit
def state_constraint(x, t):
    del t
    pos = x[0:2]
    return obs_constraint(pos)


# Define filter
@jax.jit
def get_logpd(yxs: jnp.ndarray, yus: jnp.ndarray, noise_var: float):
    def step(state, carry):
        x_hat, cov_hat, logpd = state
        yx, yu = carry
        u = unnorm_control(yu)
        x = unnorm_state(yx)
        A = jax.jacfwd(dynamics, argnums=0)(x_hat, u, 0)
        B = jax.jacfwd(dynamics, argnums=1)(x_hat, u, 0)
        Q = B @ B.T * noise_var
        R = jnp.eye(nx) * noise_var
        x_pred = A @ x_hat + B @ u
        cov_pred = A @ cov_hat @ A.T + Q
        K = cov_pred @ jnp.linalg.inv(cov_pred + R)
        x_hat = x_pred + K @ (yx - x_pred)
        cov_hat = (jnp.eye(nx) - K) @ cov_pred
        y_cov_pred = cov_pred + R
        logpd += -0.5 * (
            jnp.log(2 * jnp.pi) * nx
            + jnp.linalg.slogdet(y_cov_pred)[1]
            + (x - x_pred).T @ jnp.linalg.inv(y_cov_pred) @ (x - x_pred)
        )
        return (x_hat, cov_hat, logpd), None

    cov_hat = jnp.eye(nx) * 0.0
    logpd = 0.0
    initial_state = (x0, cov_hat, logpd)

    carry = (yxs, yus)
    (_, _, logpd), _ = lax.scan(step, initial_state, carry)
    return logpd


@jax.jit
def get_logpj(yxs: jnp.ndarray, yus: jnp.ndarray):
    xs = unnorm_state(yxs)
    us = unnorm_control(yus)
    cs = jax.vmap(cost, in_axes=(0, 0, None))(xs, us, 0.0)
    return -jnp.sum(cs)


@jax.jit
def update_traj(
    yxs: jnp.ndarray, yus: jnp.ndarray, noise_var: float, eps: float, key: chex.PRNGKey
):
    logpd_grad = jax.grad(get_logpd, argnums=(0, 1))(yxs, yus, noise_var)
    logpj_grad = jax.grad(get_logpj, argnums=(0, 1))(yxs, yus)
    yx_grad = logpd_grad[0] + logpj_grad[0]*0.0
    yu_grad = logpd_grad[1] + logpj_grad[1]*0.0

    key, xkey, ukey = jax.random.split(key, 3)
    yxs = yxs + eps * yx_grad + jax.random.normal(xkey, yxs.shape) * jnp.sqrt(2 * eps)
    yus = yus + eps * yu_grad + jax.random.normal(ukey, yus.shape) * jnp.sqrt(2 * eps)
    yxs = yxs.at[-1].set(xT)

    return yxs, yus, key


@jax.jit
def update_traj_langevine(yxs, yus, noise_var, key):
    def step(state, carry):
        yxs, yus, key = state
        eps = carry
        yxs, yus, key = update_traj(yxs, yus, noise_var, eps, key)
        return (yxs, yus, key), None

    (yxs, yus, key), _ = lax.scan(step, (yxs, yus, key), eps_schedule)
    return yxs, yus, key


yxs_guess = jnp.linspace(x0, xT, T)
yus_guess = jnp.zeros((T, nu))
key, x_key, u_key = jax.random.split(key, 3)
yxs = yxs_guess + jax.random.normal(x_key, yxs_guess.shape) * jnp.sqrt(
    noise_var_schedule[0]
)
yus = yus_guess + jax.random.normal(u_key, yus_guess.shape) * jnp.sqrt(
    noise_var_schedule[0]
)

key, subkey = jax.random.split(key)

fig, ax = render_scene()
# load X and U back
# xs_opt = jnp.load("../figure/X.npy")
# us_opt = jnp.load("../figure/U.npy")


# plot them
def plot_traj(ax, xs, xs_opt=None):
    ax.clear()
    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([-2.0, 2.0])
    ax.grid(True)
    ax.set_aspect("equal")
    if xs_opt is not None:
        ax.plot(xs_opt[:, 0], xs_opt[:, 1], "b--", linewidth=2, alpha=0.5)
        ax.quiver(
            xs_opt[:, 0],
            xs_opt[:, 1],
            jnp.sin(xs_opt[:, 2]),
            jnp.cos(xs_opt[:, 2]),
            range(T + 1),
            cmap="Blues",
        )
    ax.plot(xs[:, 0], xs[:, 1], "r-", linewidth=2, alpha=0.5)
    ax.quiver(
        xs[:, 0], xs[:, 1], jnp.sin(xs[:, 2]), jnp.cos(xs[:, 2]), range(T), cmap="Reds"
    )


# run the simulation
for i in trange(Ndiff):
    yxs, yus, subkey = update_traj_langevine(yxs, yus, noise_var_schedule[i], subkey)
    xs = unnorm_state(yxs)

    plot_traj(ax, xs, xs_opt=None)
    # update graph
    plt.pause(0.01)

exit()

# Define Solver
solver_options = dict(
    method=shootsqp.SQP_METHOD.SENS,
    ddp_options={"ddp_gamma": 1e-4},
    hess="full",
    verbose=True,
    max_iter=100,
    ls_eta=0.49,
    ls_beta=0.8,
    primal_tol=1e-3,
    dual_tol=1e-3,
    stall_check="abs",
    debug=False,
)
solver = shootsqp.ShootSQP(
    nx,
    nu,
    T,
    dynamics,
    cost,
    control_bounds,
    state_constraint,
    s1_ind=s1_indices,
    **solver_options,
)

# Set initial conditions and problem parameters
# x0 = jnp.zeros((nx,))
# x0 = jnp.array([0.25, 1.75, 0., 0.])
U0 = jnp.zeros((T, nu))
X0 = None
solver.opt.proj_init = False

# Optional X0 guess (must set solver.opt.proj_init = True)
solver.opt.proj_init = True
waypoints = jnp.array([x0[:2], jnp.array([0.0, 1.0]), xT[:2]])
X0 = jnp.concatenate(
    (
        jnp.linspace(waypoints[0], waypoints[1], int(T // 2)),
        jnp.linspace(waypoints[1], waypoints[2], int(T // 2) + 2)[1:],
    )
)

# Augment with zeros
X0 = jnp.hstack((X0, jnp.zeros((T + 1, 2))))

# Run for one iteration to jit first
solver.opt.max_iter = 1
_ = solver.solve(x0, U0, X0)

# Run to completion
solver.opt.max_iter = 100
soln = solver.solve(x0, U0, X0)

print(f"itr: {soln.iterations}, obj: {soln.objective}, kkt: {soln.kkt_residuals}")

plt.rcParams.update({"font.size": 20})
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

fig, ax = render_scene()
U, X = soln.primals
# save U, X for later
jnp.save("../figure/U.npy", U)
jnp.save("../figure/X.npy", X)
ax.plot(X[:, 0], X[:, 1], "r-", linewidth=2)

for t in jnp.arange(0, solver._T + 1, 5):
    ax.arrow(
        X[t, 0],
        X[t, 1],
        0.2 * jnp.sin(X[t, 2]),
        0.2 * jnp.cos(X[t, 2]),
        width=0.05,
        color="c",
    )

# Start
ax.add_patch(plt.Circle([x0[0], x0[1]], 0.1, color="g", alpha=0.3))
# End
ax.add_patch(plt.Circle([xT[0], xT[1]], 0.1, color="r", alpha=0.3))

ax.set_aspect("equal")
plt.show()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.grid(True)
plt.plot(solver._timesteps[:-1] * dt, U, markersize=5)
ax.set_ylabel("U")
ax.set_xlabel("Time [s]")
plt.show()

import seaborn as sns

colors = sns.color_palette("tab10")

history = soln.history
history.keys()

plt.rcParams.update({"font.size": 24})
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

axs[0][0].plot(history["steplength"], color=colors[0], linewidth=2)
axs[0][0].set_title("Step size")
axs[0][0].grid(True)

axs[0][1].plot(history["obj"], color=colors[0], linewidth=2)
axs[0][1].set_title("Objective")
axs[0][1].set_yscale("log")
axs[0][1].grid(True)

axs[1][0].plot(history["min_viol"], color=colors[0], linewidth=2)
axs[1][0].set_title("Min constraint viol.")
axs[1][0].set_xlabel("Iteration")
axs[1][0].grid(True)

if "ddp_err" in history:
    axs[1][1].plot(history["ddp_err"], color=colors[0], linewidth=2)
    axs2 = axs[1][1].twinx()
    axs2.plot(history["ddp_err_grad"], color=colors[1], linewidth=2)
    axs2.set_yscale("log")
    axs[1][1].set_title("DDP errors")
    axs[1][1].set_xlabel("Iteration")
    axs[1][1].grid(True)

plt.tight_layout()
plt.show()
