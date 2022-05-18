import numpy as np
import torch
from torch import Tensor

def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

    return res

def quat_mul_tensor(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = torch.cat([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

    return res

def quat_inv(q):
    """
    Inverts a tensor of quaternions

    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    res = np.asarray([1, -1, -1, -1], dtype=np.float32) * q
    return res

def quat_inv_tensor(q: Tensor):
    """
    Inverts a tensor of quaternions

    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    res = torch.Tensor([1, -1, -1, -1]).to(q.device) * q
    return res


def quat_norm(q: Tensor) -> Tensor:
    """Obtains the norm of a quaternion

    Args:
        q (Tensor): Tensor of Quaternions [..., 4]

    Returns:
        Tensor: Norm of the quaternions [..., 1]
    """

    return torch.sqrt(
        q[..., 0:1] * q[..., 0:1] +
        q[..., 1:2] * q[..., 1:2] +
        q[..., 2:3] * q[..., 2:3] +
        q[..., 3:4] * q[..., 3:4]
    )

def quat_angle(q: Tensor) -> Tensor:
    """Obtains the angle phi of a quaternion

    a = ||q|| cos(phi)
    phi = acos(a / ||q||)

    Args:
        q (Tensor): Tensor of quaternions. [..., 4]
    
    Returns:
        Tensor: Tensor of Phis. [..., 1]
    """
    return torch.acos(q[..., 0:1] / (quat_norm(q) + 1e-8))

def quat_unit_vector(q: Tensor) -> Tensor:
    """Returns the unit vector of the quaternions.

    q = a + v
    v = n ||v|| = n ||q|| sin(phi)
    n = v / (||q|| sin(phi))
    
    Args:
        q (Tensor): Input quaternions. [..., 4]
    
    Returns:
        Tensor: Unit vectors. [..., 3]
    """
    return q[..., 1:] / (quat_norm(q) * torch.sin(quat_angle(q)) + 1e-8)

def quat_exp(q: Tensor, x: float) -> Tensor:
    """Performs quaternion exponentiation.

    q^x = ||q||^x . (cos(x . phi) + n . sin(x . phi))

    Args:
        q (Tensor): Input quaternions. [..., 4]
        x (float): Power to raise.
    
    Returns:
        Tensor: Output quaternions. [..., 4]
    """
    norm = quat_norm(q)
    phi = quat_angle(q)
    n = quat_unit_vector(q)

    # print("norm", norm)

    # print("phi", phi)

    # print("n", n)

    x_phi = x * phi

    return torch.pow(norm, x) * torch.cat([torch.cos(x_phi), n * torch.sin(x_phi)], dim = -1)

def quat_fk(lrot, lpos, parents):
    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations

    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[..., i:i+1, :]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lrot[..., i:i+1, :]))

    res = np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
    return res

def quat_fk_tensor(lrot, lpos, parents):
    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations

    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec_tensor(gr[parents[i]], lpos[..., i:i+1, :]) + gp[parents[i]])
        gr.append(quat_mul_tensor(gr[parents[i]], lrot[..., i:i+1, :]))

    return torch.cat(gr, axis=-2), torch.cat(gp, axis=-2)

def quat_ik(grot, gpos, parents):
    """
    Performs Inverse Kinematics (IK) on global quaternions and global positions to retrieve local representations

    :param grot: tensor of global quaternions with shape (..., Nb of joints, 4)
    :param gpos: tensor of global positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of local quaternion, local positions
    """
    res = [
        np.concatenate([
            grot[..., :1, :],
            quat_mul(quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
        ], axis=-2),
        np.concatenate([
            gpos[..., :1, :],
            quat_mul_vec(
                quat_inv(grot[..., parents[1:], :]),
                gpos[..., 1:, :] - gpos[..., parents[1:], :]),
        ], axis=-2)
    ]

    return res

def quat_ik_tensor(grot, gpos, parents):
    """
    Performs Inverse Kinematics (IK) on global quaternions and global positions to retrieve local representations

    :param grot: tensor of global quaternions with shape (..., Nb of joints, 4)
    :param gpos: tensor of global positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of local quaternion, local positions
    """
    res = [
        torch.cat([
            grot[..., :1, :],
            quat_mul_tensor(quat_inv_tensor(grot[..., parents[1:], :]), grot[..., 1:, :]),
        ], dim=-2),
        torch.cat([
            gpos[..., :1, :],
            quat_mul_vec_tensor(
                quat_inv_tensor(grot[..., parents[1:], :]),
                gpos[..., 1:, :] - gpos[..., parents[1:], :]),
        ], dim=-2)
    ]

    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * np.cross(q[..., 1:], x)
    res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)

    return res

def quat_mul_vec_tensor(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * torch.cross(q[..., 1:], x)
    res = x + q[..., 0].unsqueeze(dim=-1) * t + torch.cross(q[..., 1:], t)

    return res

def quat_between(x, y):
    """
    Quaternion rotations between two 3D-vector arrays

    :param x: tensor of 3D vectors
    :param y: tensor of 3D vetcors
    :return: tensor of quaternions
    """
    res = np.concatenate([
        np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis] +
        np.sum(x * y, axis=-1)[..., np.newaxis],
        np.cross(x, y)], axis=-1)
    return res


def remove_quat_discontinuities(rotations):
    """

    Removing quat discontinuities on the time dimension (removing flips)

    :param rotations: Array of quaternions of shape (T, J, 4)
    :return: The processed array without quaternion inversion.
    """
    rots_inv = -rotations

    for i in range(1, rotations.shape[0]):
        # Compare dot products
        replace_mask = np.sum(rotations[i - 1: i] * rotations[i: i + 1], axis=-1) < np.sum(
            rotations[i - 1: i] * rots_inv[i: i + 1], axis=-1)
        replace_mask = replace_mask[..., np.newaxis]
        rotations[i] = replace_mask * rots_inv[i] + (1.0 - replace_mask) * rotations[i]

    return rotations