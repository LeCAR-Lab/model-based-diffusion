import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

l_block = 0.4
dt = 0.01


def dynamics(x, u):
    r_point = x[0:2]
    r_block = x[2:4]
    theta_block = x[4]
    v_point = u[0:2]

    # transform r_point into block frame
    Rrot_block = np.array(
        [
            [np.cos(theta_block), -np.sin(theta_block)],
            [np.sin(theta_block), np.cos(theta_block)],
        ]
    )
    r_point2block = np.dot(Rrot_block.T, r_point - r_block)  # r_point in block frame
    v_point2block = np.dot(Rrot_block.T, v_point)  # v_point in block frame

    # check if point is in contact with block in the next time step
    r_point2block_next = r_point2block + v_point2block * dt
    if np.linalg.norm(r_point2block_next, np.inf) < l_block / 2:
        # point is in contact with block
        d_point2block = np.linalg.norm(r_point2block)
        vec_point2block = r_point2block / d_point2block
        omega_block = np.cross(
            np.concatenate([vec_point2block, np.array([0.0])]),
            1 / (0.5 * l_block) * np.concatenate([v_point2block, np.array([0.0])]),
        )
        v_block2block = np.cross(
            omega_block, np.concatenate([r_point2block, np.array([0.0])])
        )
        v_block = np.dot(Rrot_block, v_block2block)
        w_block = omega_block[2]
    else:
        # point is not in contact with block
        v_block = np.zeros(2)
        w_block = 0.0

    return np.concatenate([v_point, v_block, np.array([w_block])])


x = np.array([0.0, 0.0, 0.1, 0.2, 0.0])
u = np.array([0.0, 1.0])
contact_eps = 1e-5
k_contact = 1.0  # 2.0 means large friction, 0.5 means small friction
r_point = x[0:2]
r_block = x[2:4]
theta_block = x[4]
v_point = u[0:2]

# transform r_point into block frame
Rrot_block = np.array(
    [
        [np.cos(theta_block), -np.sin(theta_block)],
        [np.sin(theta_block), np.cos(theta_block)],
    ]
)
r_point2block = np.dot(Rrot_block.T, r_point - r_block)  # r_point in block frame
v_point2block = np.dot(Rrot_block.T, v_point)  # v_point in block frame

# check contact mode
has_contact = (np.linalg.norm(r_point2block, np.inf) - l_block / 2) < contact_eps
theta_point2block = np.arctan2(r_point2block[1], r_point2block[0])
theta_pi = (theta_point2block + np.pi/2) % (np.pi) - np.pi/2
left_contact = (np.abs((theta_point2block) % (2*np.pi) - np.pi) < np.pi/4) & has_contact
right_contact = (np.abs(theta_point2block) < np.pi/4) & has_contact
top_contact = (np.abs(theta_point2block - np.pi/2) < np.pi/4) & has_contact
bottom_contact = (np.abs(theta_point2block + np.pi/2) < np.pi/4) & has_contact
# left_right_contact = (r_point2block[1] < l_block/2) & (r_point2block[1] > -l_block/2) & has_contact
# left_contact = (r_point2block[0] < 0) & left_right_contact
# right_contact = (r_point2block[0] > 0) & left_right_contact
# top_bottom_contact = (r_point2block[1] > 0) & (r_point2block[0] < l_block/2) & (r_point2block[0] > -l_block/2) & has_contact
# top_contact = (r_point2block[1] > 0) & top_bottom_contact
# bottom_contact = (r_point2block[1] < 0) & top_bottom_contact
next_has_contact = (
    np.linalg.norm(r_point2block + v_point2block * dt, np.inf) < l_block / 2
)

# check if point is in contact with block in the next time step
if next_has_contact:
    if left_contact or right_contact:
        vx = v_point2block[0]
        d = r_point2block[1]
        k = 1.0 / (1.0 + (0.5 * l_block) / (k_contact * d))
        v_block2block = np.array([vx * k, 0.0])
        w_block = -(1 - k) * vx / (0.5 * l_block)
    elif top_contact or bottom_contact:
        vy = v_point2block[1]
        d = r_point2block[0]
        k = 1.0 / (1.0 + (0.5 * l_block) / (k_contact * d))
        v_block2block = np.array([0.0, vy * k])
        w_block = (1 - k) * vy / (0.5 * l_block)
    v_block = np.dot(Rrot_block, v_block2block)
else:
    # point is not in contact with block
    v_block = np.zeros(2)
    w_block = 0.0

next_r_block = r_block + v_block * dt
next_theta_block = theta_block + w_block * dt
if has_contact:
    if left_contact:
        w_point2block = v_point2block[1] / (0.5 * l_block)
    elif right_contact:
        w_point2block = -v_point2block[1] / (0.5 * l_block)
    elif top_contact:
        w_point2block = -v_point2block[0] / (0.5 * l_block)
    elif bottom_contact:
        w_point2block = v_point2block[0] / (0.5 * l_block)
    next_theta_point2block = (w_point2block * dt + np.pi) % (2 * np.pi) - np.pi
    # check which side of the block the point is on
    if np.abs(next_theta_point2block) < np.pi / 2: # right contact
        next_r_point2block = np.array([1.0, np.sin(next_theta_point2block)]) * l_block / 2
    elif np.abs((next_theta_point2block) % (2*np.pi) - np.pi) < np.pi/2:
        next_r_point2block = np.array([-1.0, np.sin(next_theta_point2block)]) * l_block / 2
    elif np.abs(next_theta_point2block - np.pi/2) < np.pi/2:
        next_r_point2block = np.array([np.cos(next_theta_point2block), 1.0]) * l_block / 2
    elif np.abs(next_theta_point2block + np.pi/2) < np.pi/2:
        next_r_point2block = np.array([np.cos(next_theta_point2block), -1.0]) * l_block / 2
    Rrot_block_next = np.array(
        [
            [np.cos(next_theta_block), -np.sin(next_theta_block)],
            [np.sin(next_theta_block), np.cos(next_theta_block)],
        ]
    )

fig, ax = plt.subplots()
# vis_scene(ax, x)
# plt.show()
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
block = Rectangle(
    (r_block[0] - l_block / 2, r_block[1] - l_block / 2),
    l_block,
    l_block,
    angle=theta_block * 180 / np.pi,
)
ax.add_patch(block)
ax.plot(r_point[0], r_point[1], "ro")
ax.quiver(r_point[0], r_point[1], v_point[0], v_point[1], color="r")
ax.quiver(r_block[0], r_block[1], v_block[0], v_block[1], color="k")
print(omega_block)
ax.set_aspect("equal")
plt.show()
exit()


def vis_scene(ax, x):
    r_point = x[0:2]
    r_block = x[2:4]
    theta_block = x[4]

    ax.clear()
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)

    # draw block
    block = Rectangle(
        (r_block[0] - l_block / 2, r_block[1] - l_block / 2),
        l_block,
        l_block,
        angle=theta_block * 180 / np.pi,
    )
    ax.add_patch(block)

    # draw point
    ax.plot(r_point[0], r_point[1], "ro")

    ax.set_aspect("equal")
