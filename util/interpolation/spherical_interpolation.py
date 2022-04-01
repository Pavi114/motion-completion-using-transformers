import torch
from torch import LongTensor, Tensor

from util.quaternions import quat_exp, quat_inv_tensor, quat_mul_tensor

def quat_slerp(q1: Tensor, q2: Tensor, n: int, dim: int) -> Tensor:
    """Perform slerp on two quaternions to get n interpolated values.

    Slerp(q_1, q_2, t) = q_1 . (q_1^(-1) . q_2)^t

    Args:
        q1 (Tensor): Quaternion 1 [..., 4]
        q2 (Tensor): Quaternion 2 [..., 4]
        n (int): Number of inbetween values
        dim (int): Dimension to index.
    
    Returns:
        Tensor: Spherical Interpolated Tensor [..., n, 4]
    """
    q1_inv  = quat_inv_tensor(q1)

    # print("q1_inv", q1_inv)
    
    q = quat_mul_tensor(q1_inv, q2)

    # print("q", q)

    quats = []

    for i in range(n):
        t = (i + 1) / (n + 1)
        # print("quat_exp(q, t)", quat_exp(q, t))
        quats.append(quat_mul_tensor(q1, quat_exp(q, t)))

    return torch.cat(quats, dim)

def spherical_interpolation(x: Tensor, dim: int, fixed_points: LongTensor) -> Tensor:
    """Perform spherical interpolation fixed_points on a tensor

    This function accepts a tensor and a list of fixed indices.
    The fixed indices are preserved as-is and the positions in
    between are filled with pherical interpolation values.

    TODO: Optimize further, maybe use cuda, parallelize?

    Args:
        x (Tensor): Input tensor to interpolate. [..., N, ...].
        dim (int): Dimension to index.
        fixed_points (LongTensor): List of fixed indices. [i: 0 <= i < N]
                                    First and Last Indices MUST BE 0 and N - 1

    Returns:
        Tensor: Spherical Interpolated Tensor
    """
    fixed_values = x.index_select(dim, fixed_points)

    index_tensor = torch.arange(len(fixed_points)).unsqueeze(dim=1).to(fixed_points.device)

    xi = []

    for i in range(len(fixed_points) - 1):
        n = fixed_points[i + 1] - fixed_points[i] - 1

        d_range = quat_slerp(fixed_values.index_select(dim, index_tensor[i]), fixed_values.index_select(dim, index_tensor[i + 1]), n, dim)

        xi.append(fixed_values.index_select(dim, index_tensor[i]))
        xi.append(d_range)

    xi.append(fixed_values.index_select(dim, index_tensor[fixed_values.shape[dim] - 1]))

    return torch.cat(xi, dim=dim)