import jax

from jax import config

config.update("jax_enable_x64", True)

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import chex
from flax import struct
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pylab as plt
import numpy as np
import pprint
import optax
import scipy as sp
import tqdm
import time

from brax import envs, State
from brax.envs import Env
from brax.io import html
from optax import OptState, TransformUpdateFn
from trajax.integrators import euler
from trajax.optimizers import ilqr

Array = jax.Array
PRNGKey = chex.PRNGKey
PyTree = Any
Scalar = chex.Scalar

# f(t, x, u)
Dynamics = Callable[[Scalar, PyTree, Array], PyTree]

# c(t, x, u)
StageCost = Callable[[Scalar, PyTree, Array], float]
# c(t, x)
TerminalCost = Callable[[Scalar, PyTree], float]


@dataclass
class Cost:
    stage_cost: StageCost
    terminal_cost: TerminalCost


def rollout(dynamics: Dynamics, U: Array, x0: PyTree) -> PyTree:
    """Unrolls `X[t+1] = dynamics(t, X[t], U[t])`, where `X[0] = x0`."""

    def step(x, args):
        t, u = args
        x = dynamics(t, x, u)
        return x, x  # Return (carry, emitted state)

    _, X = jax.lax.scan(step, x0, (jnp.arange(len(U)), U))
    return X


def trajectory_cost(cost: Cost, U: Array, x0: PyTree, X: PyTree) -> float:
    T = len(U)
    time_steps = jnp.arange(T)
    X = jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate((a[None, :], b), axis=0), x0, X)
    stage_cost = jnp.sum(jax.vmap(cost.stage_cost)(
        time_steps, jax.tree_util.tree_map(lambda leaf: leaf[:-1], X), U))
    terminal_cost = cost.terminal_cost(
        T, jax.tree_util.tree_map(lambda leaf: leaf[-1], X))
    return stage_cost + terminal_cost


def objective(
    dynamics: Dynamics, cost: Cost, U: Array, x0: PyTree
) -> float:
    X = rollout(dynamics, U, x0)
    return trajectory_cost(cost, U, x0, X)


def pytree_block_until_ready(tree: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), tree)


def print_jit_and_eval_times(f, *, args=(), kwargs={}, name='', num_steps=5):
    start_time = time.time()
    pytree_block_until_ready(f(*args, **kwargs))
    jit_plus_eval_time = time.time() - start_time

    start_time = time.time()
    for _ in range(num_steps):
        pytree_block_until_ready(f(*args, **kwargs))
    eval_time = (time.time() - start_time) / num_steps
    print(
        f"{name}, jit_time={jit_plus_eval_time - eval_time:.3f} (s), eval_time={eval_time:.3f} (s)")
    
def state_to_gen_coords(state: State) -> Array:
    return jnp.concatenate((state.q, state.qd))


def gen_coords_to_state(env: Env, coords: Array) -> State:
    q, qd = jnp.split(coords, [env.sys.q_size()])
    return env.pipeline_init(q, qd)


def brax_dynamics(
        env: Env, t: Scalar, x: State, u: Array) -> Array:
    return env.pipeline_step(x, u)

reacher_env = envs.get_environment(
    env_name='reacher',    # 'reacher', 'ant', ...
    backend='positional',  # 'generalized', 'positional', ...
)

dynamics = partial(brax_dynamics, reacher_env)
trajectory = jax.jit(partial(rollout, dynamics))
coords_to_state = jax.jit(partial(gen_coords_to_state, reacher_env))

@struct.dataclass
class ReacherCostParams:
    # stage cost weights
    stage_pos: float = 10.0
    stage_vel: float = 0.001
    stage_act: float = 0.01

    # terminal cost weights
    term_pos: float = 100.0
    term_vel: float = 10.0

def reacher_stage_cost(
    t: Scalar, x: State, u: Array, *, params: ReacherCostParams
) -> float:
    obs = reacher_env._get_obs(x)  # get access to goal
    state_cost = 0.0

    # distance to goal
    state_cost += params.stage_pos * jnp.sum(obs[8:10] ** 2)

    # angular velocity of arm
    state_cost += params.stage_vel * jnp.sum(obs[6:8] ** 2)

    # actuation penalty
    act_cost = params.stage_act * jnp.sum(u ** 2)
    return state_cost + act_cost


def reacher_terminal_cost(
    t: Scalar, x: State, *, params: ReacherCostParams
) -> float:
    obs = reacher_env._get_obs(x)
    state_cost = 0.0

    # distance to goal
    state_cost += params.term_pos * jnp.sum(obs[8:10] ** 2)

    # angular velocity of arm
    state_cost += params.term_vel * jnp.sum(obs[6:8] ** 2)
    return state_cost

@jax.jit
def value_and_grad_obj(
    U: Array, x0: PyTree, params: ReacherCostParams
) -> tuple[float, Array]:
    cost = Cost(
        stage_cost=partial(reacher_stage_cost, params=params),
        terminal_cost=partial(reacher_terminal_cost, params=params))
    obj = partial(objective, dynamics, cost)
    return jax.value_and_grad(obj)(U, x0)

def gradient_step(
    update: TransformUpdateFn,
    U: Array,
    opt_state: OptState,
    x0: Array,
    params: ReacherCostParams
) -> tuple[float, Array, Array, OptState]:
    v, g = value_and_grad_obj(U, x0, params)
    U, opt_state = optax_step(update, U, opt_state, g)
    return v, g, U, opt_state


def optax_step(
    update: TransformUpdateFn,
    params: Array,
    opt_state: OptState,
    gradient: Array,
):
    updates, opt_state = update(gradient, opt_state, params)
    # params += updates
    params = optax.apply_updates(params, updates)
    return params, opt_state

opt = optax.adam(learning_rate=5e-2)  # or optax.sgd, ...
step = jax.jit(partial(gradient_step, opt.update))

T = 40
x0 = reacher_env.pipeline_init(
    jnp.array([0, 0, -0.05, -0.20]),
    
    # other initial conditions to try:
    # jnp.array([0, 0, -0.05, -0.15]),
    # jnp.array([0, 0, -0.1, -0.15]),
    # reacher_env.sys.init_q,
    
    jnp.zeros((4,))
)
U = jax.random.uniform(
    jax.random.PRNGKey(0),
    minval=-0.1,
    maxval=0.1,
    shape=(T, reacher_env.action_size),
)
params = ReacherCostParams(term_pos=1000.0, term_vel=500.0)

print_jit_and_eval_times(
    step, 
    args=(U, opt.init(U), x0, params),
    name='step')

opt_state = opt.init(U)
values = []
grad_norms = []
for iteration in tqdm.tqdm(range(500)):
    v, g, U, opt_state = step(U, opt_state, x0, params)
    values.append(v)
    grad_norms.append(np.linalg.norm(g, ord=np.inf))
    
plt.figure(figsize=(6, 4))

plt.subplot(1, 2, 1)
plt.semilogy(values)
plt.xlabel('iteration')
plt.ylabel('obj')
plt.title(f'final obj={values[-1]:.3f}')

plt.subplot(1, 2, 2)
plt.semilogy(grad_norms)
plt.xlabel('iteration')
plt.ylabel('norm(grad)')
plt.title(f'final norm(grad)={grad_norms[-1]:.3f}')

plt.suptitle('reacher traj opt (Adam)')

plt.tight_layout()
plt.show()

X = trajectory(U, x0)
X = jax.tree_util.tree_map(
    lambda a, b: jnp.concatenate((a[None, :], b), axis=0), x0, X)
with open('../figure/temp_optax.html', 'w') as fp:
    # we replace the `dt` value to 0.1 so that the visualization runs at a reasonable speed.
    fp.write(html.render(reacher_env.sys.replace(dt=0.1),
                         [coords_to_state(x) for x in jax.vmap(state_to_gen_coords)(X)]))