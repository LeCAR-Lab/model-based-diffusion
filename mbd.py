import jax
import chex
from jax import lax
from jax import numpy as jnp
from flax import struct
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import pandas as pd
import os          

# initialize tensorboard
writer = SummaryWriter()

# global parameters
obstacle = "umaze"  # sphere, umaze, square, wall, control
task = "drone"  # point, drone, pendulum
if task == 'pendulum':
    obstacle = 'control'
    print("task is pendulum, obstacle should be control")
# global static parameters
n_state: int = {"point": 4, "drone": 6, "pendulum": 2}[task]
n_action: int = {"point": 2, "drone": 2, "pendulum": 1}[task]
horizon: int = 50
diffuse_step = 50

batch_size = 256    
saved_batch_size = 2

# schedule langevin episilon
diffuse_substeps = 800
langevin_eps_schedule = jnp.linspace(10.0, 0.005, diffuse_substeps) * 2e-6
# if obstacle == "umaze":
#     langevin_eps_schedule = (
#         langevin_eps_schedule * 0.3
#     )  # NOTE: umaze needs smaller step size

# schedule global noise (perturbation noise)
# noise_var_init = 1e-2
noise_var_init = 1.0
noise_var_final = 2e-3
# noise_var_init = 1e-1
# noise_var_final = 1e-1
# plan in exponential space
scale = 8.0
noise_var_schedule = jnp.exp(jnp.linspace(scale, 0.0, diffuse_step)) / jnp.exp(scale)
noise_var_schedule = (
    noise_var_schedule * (noise_var_init - noise_var_final) + noise_var_final
)
# noise_std_schedule = jnp.linspace(noise_std_init, noise_std_final, diffuse_step)


def default_array(array):
    return struct.field(default_factory=lambda: jnp.array(array))


@struct.dataclass
class Params:
    # environment parameters
    dt: float = 0.1
    r_obs: float = 0.5
    init_state: jnp.ndarray = default_array(
        {
            "point": [-1.5, 0.0, 0.0, 0.0], 
            "drone": [-1.5, 0.0, jnp.pi/2, 0.0, 0.0, 0.0], 
            "pendulum": [0.0, 0.0]
        }[task]
    )
    goal_state: jnp.ndarray = default_array(
        {
            "point": [2.0, 0.0, 0.0, 0.0], 
            "drone": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            "pendulum": [jnp.pi, 0.0]
        }[task]
    )

    # diffuser parameters
    noise_var: float = 1.0
    langevin_eps: float = 1.0
    dyn_scale: float = 1.0
    reward_scale: float = 1.0
    barrier_scale: float = 2500.0
    final_scale: float = 1.0


def get_A_point(x: jnp.ndarray, params: Params) -> jnp.ndarray:
    return jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ) * params.dt + jnp.eye(n_state)


def get_B_point(x: jnp.ndarray, params: Params) -> jnp.ndarray:
    return (
        jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        * params.dt
    )


def get_A_drone(x: jnp.ndarray, params: Params) -> jnp.ndarray:
    return jnp.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ) * params.dt + jnp.eye(n_state)


def get_B_drone(x: jnp.ndarray, params: Params) -> jnp.ndarray:
    return (
        jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [jnp.cos(x[2])*3.0, 0.0],
                [jnp.sin(x[2])*3.0, 0.0],
                [0.0, 30.0],
            ]
        )
        * params.dt
    )

def get_A_pendulum(x: jnp.ndarray, params: Params) -> jnp.ndarray:
    return jnp.array(
        [
            [0.0, 2.0],
            [0.0, 0.0],
        ]
    ) * params.dt + jnp.eye(n_state)

def get_B_pendulum(x: jnp.ndarray, params: Params) -> jnp.ndarray:
    return jnp.array(
        [
            [0.0],
            [2.0],
        ]
    ) * params.dt


get_A = {"point": get_A_point, "drone": get_A_drone, "pendulum": get_A_pendulum}[task]
get_B = {"point": get_B_point, "drone": get_B_drone, "pendulum": get_B_pendulum}[task]


def rollout(x0: jnp.ndarray, u: jnp.ndarray, params: Params) -> jnp.ndarray:
    def f(x, u):
        return get_A(x, params) @ x + get_B(x, params) @ u, x

    _, x_seq = lax.scan(f, x0, u)
    return jnp.concatenate([x0[None, :], x_seq], axis=0)


Q = {
    "point": jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0])),
    "drone": jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 0.01])),
    "pendulum": None, 
}[task]
R = {
    "point": jnp.eye(n_action) * 0.1,
    "drone": jnp.eye(n_action) * 0.1,
    "pendulum": None, 
}[task]


def get_reward_navigation(
    x_traj: jnp.ndarray,
    u_traj: jnp.ndarray,
    params: Params,
) -> jnp.ndarray:
    def get_running_cost(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        x_err = x - params.goal_state
        x_err = jnp.concatenate([x_err[:2], x_err[3:]])
        # return (x_err @ Q @ x_err + u @ R @ u) / 2.0
        return (u @ R @ u) / 2.0

    running_cost = jax.vmap(get_running_cost)(x_traj, u_traj).sum()
    return -running_cost.sum()

def get_reward_pendulum(
    x_traj: jnp.ndarray,
    u_traj: jnp.ndarray,
    params: Params,
) -> jnp.ndarray:
    def get_running_cost(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        theta = x[0]
        theta_dot = x[1]
        return jnp.cos(theta) + 0.1 * theta_dot ** 2 + 0.01 * u ** 2

    running_cost = jax.vmap(get_running_cost)(x_traj, u_traj).sum()
    return -running_cost.sum()

get_reward = {
    "point": get_reward_navigation,
    "drone": get_reward_navigation,
    "pendulum": get_reward_pendulum,
}[task]

def get_final_constraint_navigation(x_traj: jnp.ndarray, params: Params) -> jnp.ndarray:
    final_err = x_traj[-1] - params.goal_state
    final_cost = final_err @ Q @ final_err
    return -final_cost

def get_final_constraint_drone(x_traj: jnp.ndarray, params: Params) -> jnp.ndarray:
    final_err = x_traj[-1] - params.goal_state
    final_err = jnp.concatenate([final_err[:2], final_err[3:]])
    final_cost = final_err @ Q @ final_err
    return -final_cost


def get_final_constraint_pendulum(x_traj: jnp.ndarray, params: Params) -> jnp.ndarray:
    return -jnp.cos(x_traj[-1, 0])-1.0

get_final_constraint = {
    "point": get_final_constraint_navigation,
    "drone": get_final_constraint_drone,
    "pendulum": get_final_constraint_pendulum,
}[task]

def x_traj_interpolate(x_traj: jnp.ndarray, interp_ratio: int) -> jnp.ndarray:
    linear_ratio = jnp.linspace(0, 1, interp_ratio)
    x_traj_interpolated = jnp.zeros((interp_ratio, x_traj.shape[0]-1, x_traj.shape[1]))
    for i in range(interp_ratio):
        x_traj_interpolated = x_traj_interpolated.at[i,:,:].set(x_traj[:-1] * linear_ratio[i] + x_traj[1:] * (1 - linear_ratio[i]))
    # print(x_traj_interpolated[:,:,0])
    return x_traj_interpolated

def get_barrier_sphere(x_traj: jnp.ndarray, u_traj: jnp.ndarray, params: Params) -> jnp.ndarray:
    def get_barrier_cost(x: jnp.ndarray) -> jnp.ndarray:
        dist2center = jnp.linalg.norm(x[:2])
        return jnp.clip((params.r_obs - dist2center), 0.0, params.r_obs) ** 2 / (
            params.r_obs**2
        )

    barrier_cost = jax.vmap(get_barrier_cost)(x_traj).sum()
    return -barrier_cost


def get_barrier_square(x_traj: jnp.ndarray, u_traj: jnp.ndarray, params: Params) -> jnp.ndarray:
    def get_barrier_cost(x: jnp.ndarray) -> jnp.ndarray:
        dist2center = jnp.linalg.norm(x[:2], ord=jnp.inf)
        return jnp.clip((params.r_obs - dist2center), 0.0, params.r_obs) ** 2 / (
            params.r_obs**2
        )

    barrier_cost = jax.vmap(get_barrier_cost)(x_traj).sum()
    return -barrier_cost


def get_barrier_wall(x_traj: jnp.ndarray, u_traj: jnp.ndarray, params: Params) -> jnp.ndarray:
    def get_barrier_cost(x: jnp.ndarray) -> jnp.ndarray:
        dist2center_normed = jnp.linalg.norm(x[:2] / jnp.array([0.2, 0.5]), ord=jnp.inf)
        return jnp.clip((1.0 - dist2center_normed), 0.0, 1.0) ** 2 * (0.2**2)

    barrier_cost = jax.vmap(get_barrier_cost)(x_traj).sum()
    return -barrier_cost


def get_barrier_umaze(x_traj: jnp.ndarray, u_traj: jnp.ndarray, params: Params) -> jnp.ndarray:
    half_width = 0.5
    keypoints = jnp.array([[-1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [-1.0, -1.0]])

    def get_barrier_cost(x: jnp.ndarray) -> jnp.ndarray:
        q = x[:2]
        dist2points = jnp.linalg.norm(q - keypoints, axis=1, ord=jnp.inf)
        dist1 = (
            (q[0] < -1.0) * dist2points[0]
            + (q[0] > 0.0) * dist2points[1]
            + (q[0] >= -1.0) * (q[0] <= 0.0) * jnp.abs(q[1] - 1.0)
        )
        dist2 = (
            (q[1] < -1.0) * dist2points[2]
            + (q[1] > 1.0) * dist2points[1]
            + (q[1] >= -1.0) * (q[1] <= 1.0) * jnp.abs(q[0])
        )
        dist3 = (
            (q[0] < -1.0) * dist2points[3]
            + (q[0] > 0.0) * dist2points[2]
            + (q[0] >= -1.0) * (q[0] <= 0.0) * jnp.abs(q[1] + 1.0)
        )
        dist2mazecenter = jnp.min(jnp.array([dist1, dist2, dist3]))
        return jnp.clip((half_width - dist2mazecenter), 0.0, half_width) ** 2
    x_inter = x_traj_interpolate(x_traj, 20)
    # print(x_inter[1])
    x_inter = x_inter.reshape(-1, x_traj.shape[1])
    barrier_cost = jax.vmap(get_barrier_cost)(x_inter).sum()/20
    return -barrier_cost


def get_barrier_rect(x_traj: jnp.ndarray, u_traj: jnp.ndarray, params: Params) -> jnp.ndarray:
    rect1_center = jnp.array([0.0, 0.0])
    rect1_half = jnp.array([0.2, 0.5])

    # calculate the distance to the edge of the rectangle\n
    def get_barrier_cost(x: jnp.ndarray) -> jnp.ndarray:
        x = x[:2]
        x_centered = jnp.clip(jnp.abs(x - rect1_center), jnp.zeros(2), rect1_half)
        dx = rect1_half - x_centered
        return jnp.minimum(dx[0], dx[1]) ** 2

    barrier_cost = jax.vmap(get_barrier_cost)(x_traj).sum()
    return -barrier_cost

def get_barrier_control(x_traj: jnp.ndarray, u_traj: jnp.ndarray, params: Params) -> jnp.ndarray:
    def get_barrier_cost(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(jnp.abs(u)-1.0, 0.0).sum()

    barrier_cost = jax.vmap(get_barrier_cost)(x_traj, u_traj).sum()
    return -barrier_cost

get_barrier = {
    "sphere": get_barrier_sphere,
    "square": get_barrier_square,
    "wall": get_barrier_wall,
    "umaze": get_barrier_umaze,
    "control": get_barrier_control,
}[obstacle]


def get_logpd_scan(
    x_traj: jnp.ndarray, u_traj: jnp.ndarray, params: Params
) -> jnp.ndarray:
    def step(state, input):
        x_hat, cov_hat, logpd = state
        u_prev, x_current = input
        A = get_A(x_hat, params)
        B = get_B(x_hat, params)
        Q = B @ B.T * params.noise_var
        R = jnp.eye(n_state) * params.noise_var
        x_pred = A @ x_hat + B @ u_prev
        cov_pred = A @ cov_hat @ A.T + Q
        K = cov_pred @ jnp.linalg.inv(cov_pred + R)
        x_hat = x_pred + K @ (x_current - x_pred)
        cov_hat = (jnp.eye(n_state) - K) @ cov_pred
        y_cov_pred = cov_pred + R
        logpd += -0.5 * (
            jnp.log(2 * jnp.pi) * n_state
            + jnp.linalg.slogdet(y_cov_pred)[1]
            + (x_current - x_pred).T @ jnp.linalg.inv(y_cov_pred) @ (x_current - x_pred)
        )
        return (x_hat, cov_hat, logpd), None

    x_hat = x_traj[0]
    cov_hat = jnp.eye(n_state) * 0.0
    logpd = 0.0
    initial_state = (x_hat, cov_hat, logpd)

    inputs = (u_traj, x_traj)
    (x_hat, cov_hat, logpd), _ = lax.scan(step, initial_state, inputs)
    return logpd

reward_grad = jax.grad(get_reward, argnums=[0, 1])
logpd_grad = jax.grad(get_logpd_scan, argnums=[0, 1])
barrier_grad = jax.grad(get_barrier, argnums=[0, 1])
final_grad = jax.grad(get_final_constraint, argnums=0)

def update_traj(
    x_traj: jnp.ndarray,
    u_traj: jnp.ndarray,
    params: Params,
    rng: chex.PRNGKey,
    clip_bound: float = 2.0,
) -> jnp.ndarray:
    reward_grad_x, reward_grad_u = reward_grad(x_traj, u_traj, params)

    logpd_grad_x, logpd_grad_u = logpd_grad(x_traj, u_traj, params)

    barrier_grad_x, barrier_grad_u = barrier_grad(x_traj, u_traj, params)

    final_grad_x = final_grad(x_traj, params)

    grad_x = (
        logpd_grad_x * params.dyn_scale
        + reward_grad_x * params.reward_scale
        + barrier_grad_x * params.barrier_scale
        + final_grad_x * params.final_scale
    )
    grad_x = jnp.clip(grad_x, -clip_bound, clip_bound)
    grad_u = logpd_grad_u * params.dyn_scale + reward_grad_u * params.reward_scale + barrier_grad_u * params.barrier_scale
    grad_u = jnp.clip(grad_u, -clip_bound, clip_bound)
    eps = params.langevin_eps
    rng, rng_x, rng_u = jax.random.split(rng, 3)

    x_traj_new = (
        x_traj
        + eps * grad_x
        + 0.1 *  jnp.sqrt(2 * eps) * jax.random.normal(rng_x, grad_x.shape)
    )
    x_traj_new = x_traj_new.at[0].set(
        params.init_state
    )  # NOTE: do not add noise to the initial state
    # x_traj_new = x_traj_new.at[-1].set(
    #     params.goal_state
    # )  # NOTE: do not add noise to the final state

    u_traj_new = (
        u_traj
        + eps * grad_u
        + 0.1 * jnp.sqrt(2 * eps) * jax.random.normal(rng_u, grad_u.shape)
    )

    return x_traj_new, u_traj_new

def vis_traj(x_traj, x_traj_real, filename):
    fig, (ax,ax2)= plt.subplots(1, 2)
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
        if task == 'point' or task == 'drone':
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
        elif task == 'pendulum':
            ax.plot(
                x_traj[j, :, 0],
                "r",
                alpha=1.0,
            )
            ax.plot(
                x_traj_real[j, :, 0],
                "b--",
            )
    ax.grid()
    if task == 'point' or task == 'drone':
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_aspect("equal", adjustable="box")
        # plot star at [1, 0]
        ax.plot(params.goal_state[0],params.goal_state[1], "r*", markersize=16)
    # set title
    ax.set_title("Trajectory")
    # save figure to file
    ## Add a subplot
    ## Set ax2 has the saax2.axis('square')
    ax2.set_aspect('auto', adjustable='box')
    ax2.plot(x_traj[0, :, 0], label='x')
    ax2.plot(x_traj[0, :, 1], label='y')
    ax2.legend()
    ax2.set_title("Trajectory")
    plt.savefig(f"figure/{filename}.png")
    ax2.clear()
    ax.clear()
    plt.close(fig)

# init params
params = Params()
default_params = Params()
rng = jax.random.PRNGKey(1)

# test out dynamics with pid controller
if task == 'drone':
    key_points = jnp.array([
        [-2.0, 0.5], 
        [-2.0, 2.0], 
        [1.0, 2.0], 
        [1.0, 0.0]
    ])
    def pid_control(x, x_goal):
        # x_goal = jnp.array([1.0, 0.0])
        x_err = x_goal - x[:2]
        fd = x_err*3.0 - x[3:5]*4.0
        # fd = x_err*10.0 - x[3:5]*12.0
        theta = jnp.arctan2(fd[1], fd[0])
        theta_err = theta - x[2]
        theta_err = (theta_err + jnp.pi) % (2 * jnp.pi) - jnp.pi
        # calculate thrust by project fd into the direction of drone
        n = jnp.array([jnp.cos(x[2]), jnp.sin(x[2])])
        thrust = jnp.dot(fd, n)
        thrust = jnp.clip(thrust, -1.0, 1.0)
        torque = jnp.clip(theta_err*1.0-x[5]*0.3, -1.0, 1.0)
        return jnp.array([thrust, torque])
    desired_point_idx = 0
    x = params.init_state
    xs = []
    for t in range(horizon):
        u = pid_control(x, key_points[desired_point_idx])
        x = get_A(x, params) @ x + get_B(x, params) @ u
        if jnp.linalg.norm(x[:2] - key_points[desired_point_idx]) < 0.2:
            desired_point_idx += 1
        xs.append(x)
    x_traj_pid = jnp.array(xs)
    vis_traj(x_traj_pid[None, :], x_traj_pid[None, :], "pid")

# exit()

# init trajectory
rng, rng_x, rng_u = jax.random.split(rng, 3)
"""
RRT trajectory initialization 
# generate a line between start and goal
rrt_points = jnp.array([
    [1.0, 0.0], 
    [0.6925907100774582, 1.9504409847378015], 
    [-0.7189431117251508, 1.78907564525834],
    [-1.7817912706075743, 1.322019216390491], 
    [-1.5637527104973108, 0.7429417440998329],
    [-1.0, 0.0]
])[::-1]
# linear interpolate
points_per_segment = horizon // (rrt_points.shape[0] - 1)
xs, vs = [], []
for i in range(rrt_points.shape[0] - 1):
    xs.append(jnp.arange(0, points_per_segment)[:, None] / points_per_segment * (rrt_points[i+1] - rrt_points[i]) + rrt_points[i])
    vs.append(jnp.ones((points_per_segment, 1)) * (rrt_points[i+1] - rrt_points[i]) / points_per_segment / params.dt)
xs = jnp.concatenate(xs, axis=0) 
vs = jnp.concatenate(vs, axis=0)
x_traj_guess = jnp.concatenate([xs, vs], axis=1)
x_traj = x_traj_guess[None, :] + jax.random.normal(rng_x, (batch_size, horizon, n_state)) * jnp.array([0.2, 0.2, 0.5, 0.5])
"""
x_traj_guess = jnp.zeros((horizon, n_state))
x_traj_guess = x_traj_guess.at[:, 0].set(jnp.linspace(-1.5, 2.0, horizon))
# x_traj_guess = x_traj_guess.at[:, 1].set(jnp.ones(horizon) * 2.0)
# if task == "drone":
#     x_traj_guess = x_traj_guess.at[:, 2].set(jnp.ones(horizon) * jnp.pi)
noise_std = {
    "point": jnp.array([1.0, 2.0, 5.0, 5.0]),
    "drone": jnp.array([1.0, 2.0, 1.0, 5.0, 5.0, 1.0]),
    "pendulum": jnp.array([jnp.pi, 1.0]),
}[task]
x_traj_noise = jax.random.normal(rng_x, (batch_size, horizon, n_state)) * noise_std
if task == "point":
    x_traj = x_traj_guess[None, :] + x_traj_noise * jnp.array([1.0, 3.0, 1.0, 3.0])
elif task == "drone":
    x_traj = x_traj_guess[None, :] + x_traj_noise * jnp.array([2.0, 6.0, jnp.pi,2* 1.0, 6.0, 3.0])
elif task == "pendulum":
    x_traj = x_traj_noise * jnp.array([jnp.pi, 1.0])
u_traj = jax.random.normal(rng_u, (batch_size, horizon, n_action)) 
x_traj = x_traj.at[:, 0].set(params.init_state)

# initialize parameters
logpd_grad_x, logpd_grad_u = jax.vmap(logpd_grad, in_axes=(0,0,None))(x_traj, u_traj, params)
reward_grad_x, reward_grad_u = jax.vmap(reward_grad, in_axes=(0,0,None))(x_traj, u_traj, params)
barrier_grad_x, barrier_grad_u = jax.vmap(barrier_grad, in_axes=(0,0,None))(x_traj,u_traj, params)
final_grad_x = jax.vmap(final_grad, in_axes=(0,None))(x_traj, params)
logpd_grad_x_norm = jnp.linalg.norm(logpd_grad_x, axis=-1).mean()
reward_grad_x_norm = jnp.linalg.norm(reward_grad_x, axis=-1).mean()
barrier_grad_x_norm = jnp.linalg.norm(barrier_grad_x, axis=-1).mean()
final_grad_x_norm = jnp.linalg.norm(final_grad_x, axis=-1).mean()
params = params.replace(
    dyn_scale = 1.0, 
    reward_scale = 3e-3, 
    barrier_scale = 1500.0,
    final_scale = 150.0,
)
jax.debug.print(
    "initial dyn_scale = {x:.3f}, reward_scale = {y:.3f}, barrier_scale = {z:.3f}, final_scale = {w:.3f}",
    x=params.dyn_scale,
    y=params.reward_scale,
    z=params.barrier_scale,
    w=params.final_scale,
)
barrier_threshold = 0.009
save_infos = []
update_traj_jit = jax.jit(update_traj)
get_logpd_scan_jit = jax.jit(get_logpd_scan)
get_barrier_jit = jax.jit(get_barrier)
get_reward_jit = jax.jit(get_reward)
get_final_constraint_jit = jax.jit(get_final_constraint)
lagerangian_substeps = 10
for d_step in range(diffuse_step):
    # schedule noise_var
    barrier_goal = barrier_threshold + 0.02/50*d_step
    noise_var = noise_var_schedule[d_step]
    # if d_step > 1:
        # diffuse_substeps = diffuse_substeps * 0.8
        # diffuse_substeps = jnp.clip(diffuse_substeps, 1000, 5000)
        # start_langevin = 0.1
        # start_langevin = jnp.clip(start_langevin, 0.1, 300.0)
        # langevin_eps_schedule = jnp.linspace(start_langevin, 0.01, diffuse_substeps) * 2e-6
    for barrier_step in range(lagerangian_substeps): 
        params = params.replace(noise_var=noise_var)
        for sub_step in range(diffuse_substeps):

            # schedule langevin_eps
            langevin_eps = langevin_eps_schedule[sub_step] * (params.noise_var / 2e-5)
            
            params = params.replace(langevin_eps=langevin_eps)

            # update trajectory
            rng, rng_traj = jax.random.split(rng)
            if d_step == 0:
                x_traj, u_traj = jax.vmap(update_traj_jit, in_axes=(0, 0, None, 0))(
                    x_traj, u_traj, params, jax.random.split(rng, batch_size)
                )
            elif d_step > 0 and sub_step > 100:
                x_traj, u_traj = jax.vmap(update_traj_jit, in_axes=(0, 0, None, 0, None))(
                    x_traj, u_traj, params, jax.random.split(rng_traj, batch_size), 0.25
                )
            elif d_step > 0 and sub_step <= 100:
                _, u_traj = jax.vmap(update_traj_jit, in_axes=(0, 0, None, 0, None))(
                    x_traj, u_traj, params, jax.random.split(rng_traj, batch_size), 0.3
                )
            logpd_batch = jax.vmap(get_logpd_scan_jit, in_axes=(0, 0, None))(
                x_traj, u_traj, params
            )
            logpd = logpd_batch.mean()
            logpd_top = jnp.percentile(logpd_batch, 95)
            logp_reward_batch = jax.vmap(get_reward_jit, in_axes=(0, 0, None))(
                x_traj, u_traj, params
            )
            logp_reward = logp_reward_batch.mean()
            logp_reward_top = jnp.percentile(logp_reward_batch, 95)
            barrier_value_batch = jax.vmap(get_barrier_jit, in_axes=(0, 0, None))(x_traj, u_traj, params)
            barrier_value = barrier_value_batch.mean()
            barrier_value_top = jnp.percentile(barrier_value_batch, 95)
            final_value_batch = jax.vmap(get_final_constraint_jit, in_axes=(0, None))(x_traj, params)
            final_value = final_value_batch.mean()
            final_value_top = jnp.percentile(final_value_batch, 95)

            # tensorboard
            writer.add_scalar(
                "objective/barrier", barrier_value, d_step * diffuse_substeps + sub_step
            )
            writer.add_scalar(
                "scale/barrier_scale", params.barrier_scale, d_step * diffuse_substeps + sub_step
            )
            writer.add_scalar(
                "scale_normed/barrier_scale_normed",
                params.barrier_scale / (default_params.barrier_scale+1e-3),
                d_step * diffuse_substeps + sub_step,
            )
            writer.add_scalar("objective/dyn", logpd, d_step * diffuse_substeps + sub_step)
            writer.add_scalar(
                "scale/dyn_scale", params.dyn_scale, d_step * diffuse_substeps + sub_step
            )
            writer.add_scalar(
                "scale_normed/dyn_scale_normed",
                params.dyn_scale / (default_params.dyn_scale+1e-3),
                d_step * diffuse_substeps + sub_step,
            )
            writer.add_scalar("objective/final", final_value, d_step * diffuse_substeps + sub_step)
            writer.add_scalar(
                "scale/final_scale", params.final_scale, d_step * diffuse_substeps + sub_step
            )
            writer.add_scalar(
                "scale_normed/final_scale_normed",
                params.final_scale / (default_params.final_scale+1e-3),
                d_step * diffuse_substeps + sub_step,
            )
            writer.add_scalar("objective/reward", logp_reward, d_step * diffuse_substeps + sub_step)
            writer.add_scalar(
                "scale/reward_scale", params.reward_scale, d_step * diffuse_substeps + sub_step
            )
            writer.add_scalar(
                "scale_normed/reward_scale_normed",
                params.reward_scale / (default_params.reward_scale+1e-3),
                d_step * diffuse_substeps + sub_step,
            )
            writer.add_scalar("noise_var", noise_var, d_step * diffuse_substeps + sub_step)
            writer.add_scalar(
                "langevin_eps", langevin_eps, d_step * diffuse_substeps + sub_step
            )

            

            # rollout dynamics to get real trajectory
            x_traj_real = jax.vmap(rollout, in_axes=(0, 0, None))(
                x_traj[:, 0], u_traj, params
            )
            if sub_step % 10 == 9:
                jax.debug.print(
                    "i = {substep}/{d_step}, var = {noise:.2e}, Dyn = {x:.3e}({x1:.3e}), J = {y:.3e}({y1:.3e}), Bar = {z:.3e}({z1:.3e}), Final = {w:.3e}({w1:.3e})",
                    d_step=d_step,
                    substep=sub_step,
                    noise = noise_var,
                    x=logpd,
                    x1=params.dyn_scale,
                    y=logp_reward,
                    y1=params.reward_scale,
                    z=barrier_value,
                    z1=params.barrier_scale,
                    w=final_value,
                    w1=params.final_scale,
                )
                jax.debug.print(
                    "top_value, Dyn = {x:.3e}, J = {y:.3e}, Bar = {z:.3e}, Final = {w:.3e}",
                    x=logpd_top,
                    y=logp_reward_top,
                    z=barrier_value_top,
                    w=final_value_top,
                )
        save_infos.append([x_traj[:saved_batch_size], x_traj_real[:saved_batch_size]])
        barrier = jax.vmap(get_barrier_jit, in_axes=(0, 0, None))(x_traj, u_traj, params).mean()
        barrier_scale = params.barrier_scale + jnp.clip(
        jnp.exp((-barrier - barrier_threshold)/ 0.002) - 1.0, -100.0, 100.0)
        barrier_scale = jnp.maximum(barrier_scale, 0.0)
        dyn_scale = 1.0
        final = jax.vmap(get_final_constraint_jit, in_axes=(0, None))(x_traj, params).mean()
        final_scale = params.final_scale + jnp.clip(jnp.exp((-final-2e-2)/ 0.006) - 1.0, -5.0, 5.0)
        final_scale = jnp.maximum(final_scale, 0.0)
        # reward = jax.vmap(get_reward_jit, in_axes=(0, 0, None))(x_traj, u_traj, params).mean()
        # # schedule dynamic, reward and barrier scale
        # reward_scale = params.reward_scale + jnp.clip(
        #     jnp.exp((logpd-10.0) / 1000.0) - 1.0, -1.0, 1.0
        # )
        # reward_scale = jnp.maximum(reward_scale, 0.0)
        
        params = params.replace(
        barrier_scale=barrier_scale,
        noise_var=noise_var,
        dyn_scale=dyn_scale,
        # reward_scale=reward_scale,
        final_scale=final_scale
        )
        print('barrier goal:', -barrier_goal)
        if (-barrier)<(barrier_goal) and (-final) < 0.05:
            break 
    if d_step == 0:
        barrier_threshold = -barrier
        print('barrier_threshold:', barrier_threshold)
    # get values for dynamic, reward and barrier scale
    # logpd = jax.vmap(get_logpd_scan_jit, in_axes=(0, 0, None))(
    #     x_traj, u_traj, params
    # ).mean()
    
    # dyn_scale = jnp.maximum(dyn_scale, 0.0)
    
    # plot trajectory
    vis_traj(x_traj[:4], x_traj_real[:4], f"traj_{d_step}")

# save save_infos
x_traj_save = jnp.stack([x[0] for x in save_infos], axis=0)
x_traj_real_save = jnp.stack([x[1] for x in save_infos], axis=0)
jnp.save("figure/x_traj.npy", x_traj_save)
jnp.save("figure/x_traj_real.npy", x_traj_real_save)

# log information
log_info = {
    "d_step": 0,
    "sub_step": 0,
    "total_step": 0,
    "barrier_scale": jnp.zeros(diffuse_step * diffuse_substeps),
    "reward_scale": jnp.zeros(diffuse_step * diffuse_substeps),
    "dyn_scale": jnp.zeros(diffuse_step * diffuse_substeps),
    "final_scale": jnp.zeros(diffuse_step * diffuse_substeps),
    "barrier_scale_normed": jnp.zeros(diffuse_step * diffuse_substeps),
    "reward_scale_normed": jnp.zeros(diffuse_step * diffuse_substeps),
    "dyn_scale_normed": jnp.zeros(diffuse_step * diffuse_substeps),
    "final_scale_normed": jnp.zeros(diffuse_step * diffuse_substeps),
    "noise_var": jnp.zeros(diffuse_step * diffuse_substeps),
    "langevin_eps": jnp.zeros(diffuse_step * diffuse_substeps),
    "logpd": jnp.zeros(diffuse_step * diffuse_substeps),
    "logp_reward": jnp.zeros(diffuse_step * diffuse_substeps),
    "barrier": jnp.zeros(diffuse_step * diffuse_substeps),
}