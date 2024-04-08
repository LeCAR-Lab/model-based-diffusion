import meshcat
from meshcat import geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation, convert_frames_to_video
import numpy as np
import time
import pickle

# Create a Meshcat visualizer
vis = meshcat.Visualizer()
anim = Animation(default_framerate=50)


def origin_vec_to_transform(origin, vec, scale=1.0):
    # visualize the force with arrow
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return np.array(
            [
                [1, 0, 0, origin[0]],
                [0, 1, 0, origin[1]],
                [0, 0, 1, origin[2]],
                [0, 0, 0, 1],
            ]
        )
    vec = vec / vec_norm
    # gernerate two unit vectors perpendicular to the force vector
    if vec[0] == 0 and vec[1] == 0:
        vec_1 = np.array([1, 0, 0])
        vec_2 = np.array([0, 1, 0])
    else:
        vec_1 = np.array([vec[1], -vec[0], 0])
        vec_1 /= np.linalg.norm(vec_1)
        vec_2 = np.cross(vec, vec_1)
    rot_mat = np.eye(4)
    rot_mat[:3, 2] = vec
    rot_mat[:3, 0] = vec_1
    rot_mat[:3, 1] = vec_2
    rot_mat[:3, :3] *= vec_norm * scale
    return tf.translation_matrix(origin)


def pos_quat_to_transform(pos, quat):
    # convert quat from [x,y,z, w] to [w, x,y,z]
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    return tf.translation_matrix(pos) @ tf.quaternion_matrix(quat)


def set_frame(i, name, transform):
    # convert quat from [x,y,z, w] to [w, x,y,z]
    with anim.at_frame(vis, i) as frame:
        frame[name].set_transform(transform)


# Add a box to the scene
box = g.Box([1, 1, 1])
vis["drone"].set_object(g.StlMeshGeometry.from_file("assets/crazyflie2.stl"))
vis["drone_frame"].set_object(g.StlMeshGeometry.from_file("assets/axes.stl"))

for i in range(0, 300, 2):
    vis[f"traj{i}"].set_object(
        g.Sphere(0.01), material=g.MeshLambertMaterial(color=0x00FF00)
    )

# load state sequence from pickle and check if load is successful
file_path = "../figure/xs.npy"
xs = np.load(file_path)

# Apply the transformations according to the time sequence
for i, state in enumerate(xs):
    r = state[:3]
    q = state[3:7]
    set_frame(i, "drone", pos_quat_to_transform(r, q))
    set_frame(i, "drone_frame", pos_quat_to_transform(r, q))

vis.set_animation(anim)
time.sleep(20)