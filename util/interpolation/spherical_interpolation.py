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
    between are filled with spherical interpolation values.

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

def single_spherical_interpolation(x: Tensor, dim: int, front: int = 10, keyframe_gap: int = 30, back: int = 10) -> Tensor:
    """Perform spherical interpolation on a sequence with only one keyframe gap.

    This function accepts a tensor and the number front and back keyframes.
    The keyframes are preserved as is and the gap in the middle is filled
    with spherical interpolation values.

    Args:
        x (Tensor): Input Tensor to interpolate. [..., N, ...].
                    N = front + keyframe_gap + back
        dim (int): Dimension to index
        front (int, optional): Length of initial keyframes. Defaults to 10.
        keyframe_gap (int, optional): Length of keyframe gap. Defaults to 30.
        back (int, optional): Length of final keyframes. Defaults to 10.
    
    Returns:
        Tensor: Spherical Interpolated Tensor
    """
    # Define index tensor
    index_tensor = torch.arange(x.shape[dim]).unsqueeze(dim=1).to(x.device)

    # Extract keyframe boundaries
    first = x.index_select(dim, index_tensor[front - 1])
    last = x.index_select(dim, index_tensor[front + keyframe_gap])

    # Interpolate
    trans_sequence = quat_slerp(first, last, keyframe_gap, dim)

    # Concatenate
    x = torch.cat([
        x.index_select(dim, index_tensor[:front].squeeze()),
        trans_sequence,
        x.index_select(dim, index_tensor[front + keyframe_gap:].squeeze())
    ], dim)

    return x