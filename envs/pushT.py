import jax
from jax import numpy as jnp
import brax
from brax.envs.base import PipelineEnv, State
from brax.generalized import pipeline
from matplotlib.patches import Circle, Rectangle
from matplotlib import transforms
import matplotlib.pyplot as plt

from brax.io import mjcf


class PushT(PipelineEnv):
    def __init__(self, backend: str = "generalized"):
        self._dt = 0.02
        self._reset_count = 0
        self._step_count = 0
        sys = mjcf.loads(
            """
            <mujoco>
            <option timestep="0.02" integrator="Euler" gravity="0 0 -9.81"/>

            <worldbody>
                <body name="sphere1" pos="0.0 0 0.0">
                    <geom name="sphere1_geom" type="sphere" size="0.1" rgba="1 0 0 1" friction="1 0.001 0.0001" mass="0.5"/>
                    <joint name="sphere1_x" type="slide" axis="1 0 0" limited="true" range="-2 2"/>
                    <joint name="sphere1_y" type="slide" axis="0 1 0" limited="true" range="-2 2"/>
                </body>

                <body name="box" pos="0.0 0 0.0">
                    <geom name="box_geom" type="box" size="0.3 0.1 0.1" rgba="0 0 1 1" friction="1 0.1 0.1" mass="0.1"/>
                    <joint name="box_x" type="slide" axis="1 0 0" limited="true" range="-2 2"/>
                    <joint name="box_y" type="slide" axis="0 1 0" limited="true" range="-2 2"/>
                    <joint name="box_z_rot" type="hinge" axis="0 0 1" limited="true" range="-3.14159 3.14159"/>
                </body>
            </worldbody>

            <actuator>
                <motor name="sphere1_x_motor" joint="sphere1_x" gear="1" />
                <motor name="sphere1_y_motor" joint="sphere1_y" gear="1" />
            </actuator>
            </mujoco>
        """
        )

        super().__init__(sys, backend=backend, n_frames=10)

    def reset(self, rng: jnp.ndarray) -> State:
        




def visualize(ax, pos):
    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.add_patch(Circle((pos[0], pos[1]), 0.1, color="r"))
    rec = Rectangle((-0.3, -0.1), 0.6, 0.2, color="b")
    t = transforms.Affine2D().rotate(pos[4]).translate(pos[2], pos[3])
    rec.set_transform(t + ax.transData)
    ax.add_patch(rec)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")


fig, ax = plt.subplots()
# visualize(ax, [0, 0, 1, 0])

init_q = jnp.array([0.3, -0.2, 0.0, 0, 0.0])
state = jax.jit(pipeline.init)(scene, init_q, jnp.zeros(scene.qd_size()))

for _ in range(100):
    state = jax.jit(pipeline.step)(scene, state, jnp.array([0.0, 1.0]))
    visualize(ax, state.q)
    plt.pause(0.1)
