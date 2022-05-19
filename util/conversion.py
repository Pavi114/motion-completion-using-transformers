# This file will contain conversions like 
#   - euler_angles to quaternions
#   - quaternions to euler_angles
#   - absolute positions to angles
#   - bvh to angles
# etc. Add when necessary.

import numpy as np
import torch

from constants import DEVICE
from . import quaternions


def euler_to_quat(e, order='zyx'):
    """

    Converts from an euler representation to a quaternion representation

    :param e: euler tensor
    :param order: order of euler rotations
    :return: quaternion tensor
    """
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = angle_axis_to_quat(e[..., 0], axis[order[0]])
    q1 = angle_axis_to_quat(e[..., 1], axis[order[1]])
    q2 = angle_axis_to_quat(e[..., 2], axis[order[2]])

    return quaternions.quat_mul(q0, quaternions.quat_mul(q1, q2))


def angle_axis_to_quat(angle, axis):
    """
    Converts from and angle-axis representation to a quaternion representation

    :param angle: angles tensor
    :param axis: axis tensor
    :return: quaternion tensor
    """
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q

# Orient the data according to the las past keframe
def rotate_at_frame(X, Q, parents, n_past=10):
    """
    Re-orients the animation data according to the last frame of past context.

    :param X: tensor of local positions of shape (Batchsize, Timesteps, Joints, 3)
    :param Q: tensor of local quaternions (Batchsize, Timesteps, Joints, 4)
    :param parents: list of parents' indices
    :param n_past: number of frames in the past context
    :return: The rotated positions X and quaternions Q
    """
    # Get global quats and global poses (FK)
    global_q, global_x = quaternions.quat_fk(Q, X, parents)

    key_glob_Q = global_q[:, n_past - 1: n_past, 0:1, :]  # (B, 1, 1, 4)
    forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
                 * quaternions.quat_mul_vec(key_glob_Q, np.array([0, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :])
    forward = normalize(forward)
    yrot = normalize(quaternions.quat_between(np.array([1, 0, 0]), forward))
    new_glob_Q = quaternions.quat_mul(quaternions.quat_inv(yrot), global_q)
    new_glob_X = quaternions.quat_mul_vec(quaternions.quat_inv(yrot), global_x)

    # back to local quat-pos
    Q, X = quaternions.quat_ik(new_glob_Q, new_glob_X, parents)

    return X, Q

# Orient the data according to the las past keframe
def rotate_at_frame_tensor(X, Q, parents, n_past=10):
    """
    Re-orients the animation data according to the last frame of past context.

    :param X: tensor of local positions of shape (Batchsize, Timesteps, Joints, 3)
    :param Q: tensor of local quaternions (Batchsize, Timesteps, Joints, 4)
    :param parents: list of parents' indices
    :param n_past: number of frames in the past context
    :return: The rotated positions X and quaternions Q
    """
    # Get global quats and global poses (FK)
    global_q, global_x = quaternions.quat_fk_tensor(Q, X, parents)

    key_glob_Q = global_q[:, n_past - 1: n_past, 0:1, :]  # (B, 1, 1, 4)

    forward = torch.Tensor([1, 0, 1]).reshape((1, 1, 1, 3)) \
                 * quaternions.quat_mul_vec_tensor(key_glob_Q, torch.Tensor([0, 1, 0]).reshape((1, 1, 1, 3)))

    forward = normalize_tensor(forward)

    yrot = normalize_tensor(quaternions.quat_between_tensor(torch.Tensor([1, 0, 0]).reshape((1, 1, 1, 3)), forward))

    global_q = quaternions.quat_mul_tensor(quaternions.quat_inv_tensor(yrot), global_q)
    global_x = quaternions.quat_mul_vec_tensor(quaternions.quat_inv_tensor(yrot), global_x)

    # back to local quat-pos
    Q, X = quaternions.quat_ik_tensor(global_q, global_x, parents)

    return X, Q

def length(x, axis=-1, keepdims=True):
    """
    Computes vector norm along a tensor axis(axes)

    :param x: tensor
    :param axis: axis(axes) along which to compute the norm
    :param keepdims: indicates if the dimension(s) on axis should be kept
    :return: The length or vector of lengths.
    """
    lgth = np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims))
    return lgth

def length_tensor(x, axis=-1, keepdims=True):
    """
    Computes vector norm along a tensor axis(axes)

    :param x: tensor
    :param axis: axis(axes) along which to compute the norm
    :param keepdims: indicates if the dimension(s) on axis should be kept
    :return: The length or vector of lengths.
    """
    lgth = torch.sqrt(torch.sum(x * x, dim=axis, keepdim=keepdims))
    return lgth


def normalize(x, axis=-1, eps=1e-8):
    """
    Normalizes a tensor over some axis (axes)

    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    res = x / (length(x, axis=axis) + eps)
    return res

def normalize_tensor(x, axis=-1, eps=1e-8):
    """
    Normalizes a tensor over some axis (axes)

    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    res = x / (length_tensor(x, axis=axis) + eps)
    return res

def extract_feet_contacts(pos, lfoot_idx, rfoot_idx, velfactor=0.02):
    """
    Extracts binary tensors of feet contacts

    :param pos: tensor of global positions of shape (Timesteps, Joints, 3)
    :param lfoot_idx: indices list of left foot joints
    :param rfoot_idx: indices list of right foot joints
    :param velfactor: velocity threshold to consider a joint moving or not
    :return: binary tensors of left foot contacts and right foot contacts
    """
    lfoot_xyz = (pos[1:, lfoot_idx, :] - pos[:-1, lfoot_idx, :]) ** 2
    contacts_l = (np.sum(lfoot_xyz, axis=-1) < velfactor)

    rfoot_xyz = (pos[1:, rfoot_idx, :] - pos[:-1, rfoot_idx, :]) ** 2
    contacts_r = (np.sum(rfoot_xyz, axis=-1) < velfactor)

    # Duplicate the last frame for shape consistency
    contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
    contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)

    return contacts_l, contacts_r

