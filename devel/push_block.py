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
    Rrot_block = np.array([
        [np.cos(theta_block), -np.sin(theta_block)],
        [np.sin(theta_block), np.cos(theta_block)]
    ])
    r_point2block = np.dot(Rrot_block.T, r_point - r_block) # r_point in block frame
    v_point2block = np.dot(Rrot_block.T, v_point) # v_point in block frame

    # check if point is in contact with block in the next time step
    r_point2block_next = r_point2block + v_point2block*dt
    if np.linalg.norm(r_point2block_next, np.inf) < l_block/2:
        # point is in contact with block
        d_point2block = np.linalg.norm(r_point2block)
        vec_point2block = r_point2block / d_point2block
        omega_block = np.cross(np.concatenate([vec_point2block, np.array([0.0])]), 1/(0.5*l_block)*np.concatenate([v_point2block, np.array([0.0])]))
        v_block2block = np.cross(omega_block, np.concatenate([r_point2block, np.array([0.0])]))
        v_block = np.dot(Rrot_block, v_block2block)
        w_block = omega_block[2]
    else:
        # point is not in contact with block
        v_block = np.zeros(2)
        w_block = 0.0

    return np.concatenate([v_point, v_block, np.array([w_block])])
    
x = np.array([0.0, 0.0, 0.2, 0.2, 0.0])
u = np.array([0.0, 1.0])
r_point = x[0:2]
r_block = x[2:4]
theta_block = x[4]
v_point = u[0:2]

# transform r_point into block frame
Rrot_block = np.array([
    [np.cos(theta_block), -np.sin(theta_block)],
    [np.sin(theta_block), np.cos(theta_block)]
])
r_point2block = np.dot(Rrot_block.T, r_point - r_block) # r_point in block frame
v_point2block = np.dot(Rrot_block.T, v_point) # v_point in block frame

# check if point is in contact with block in the next time step
r_point2block_next = r_point2block + v_point2block*dt
if np.linalg.norm(r_point2block_next, np.inf) <= (l_block/2):
    # point is in contact with block
    d_point2block = np.linalg.norm(r_point2block)
    vec_point2block = r_point2block / d_point2block
    omega_block = np.cross(np.concatenate([vec_point2block, np.array([0.0])]), 1/(0.5*l_block)*np.concatenate([v_point2block, np.array([0.0])]))
    v_block2block = np.cross(omega_block, np.concatenate([r_point2block, np.array([0.0])]))
    v_block = np.dot(Rrot_block, v_block2block[0:2])
    w_block = omega_block[2]
else:
    # point is not in contact with block
    v_block = np.zeros(2)
    w_block = 0.0
fig, ax = plt.subplots()
# vis_scene(ax, x)
# plt.show()
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
block = Rectangle((r_block[0]-l_block/2, r_block[1]-l_block/2), l_block, l_block, angle=theta_block*180/np.pi)
ax.add_patch(block)
ax.plot(r_point[0], r_point[1], 'ro')
ax.quiver(r_point[0], r_point[1], v_point[0], v_point[1], color='r')
ax.quiver(r_block[0], r_block[1], v_block[0], v_block[1], color='k')
print(omega_block)
ax.set_aspect('equal')
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
    block = Rectangle((r_block[0]-l_block/2, r_block[1]-l_block/2), l_block, l_block, angle=theta_block*180/np.pi)
    ax.add_patch(block)
    
    # draw point
    ax.plot(r_point[0], r_point[1], 'ro')
    
    ax.set_aspect('equal')
