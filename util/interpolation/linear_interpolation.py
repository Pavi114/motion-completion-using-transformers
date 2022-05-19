import numpy as np
import torch
from torch import LongTensor, Tensor

def linear_interpolation(x: Tensor, dim: int, fixed_points: LongTensor) -> Tensor:
    """Perform linear interpolation fixed_points on a tensor

    This function accepts a tensor and a list of fixed indices.
    The fixed indices are preserved as-is and the positions in
    between are filled with linear interpolation values.

    TODO: Optimize further, maybe use cuda, parallelize?

    Args:
        x (Tensor): Input tensor to interpolate. [..., N, ...].
        dim (int): Dimension to index.
        fixed_points (LongTensor): List of fixed indices. [i: 0 <= i < N]
                                    First and Last Indices MUST BE 0 and N - 1

    Returns:
        Tensor: Linear Interpolated Tensor
    """
    fixed_values = x.index_select(dim, fixed_points)

    xi = []

    index_tensor = torch.arange(len(fixed_points)).unsqueeze(dim=1).to(fixed_points.device)

    for i in range(len(fixed_points) - 1):
        n = fixed_points[i + 1] - fixed_points[i]

        delta = (fixed_values.index_select(dim, index_tensor[i + 1]) - fixed_values.index_select(dim, index_tensor[i])) / n

        d_range = []

        # TODO: Optimize
        for j in range(n):
            d_range.append((fixed_values.index_select(dim, index_tensor[i]) + delta * j))

        xi.append(torch.cat(d_range, dim=dim))

    xi.append(fixed_values.index_select(dim, index_tensor[fixed_values.shape[dim] - 1]))

    return torch.cat(xi, dim=dim)

def single_linear_interpolation(x: Tensor, dim: int, front: int = 10, keyframe_gap: int = 30, back: int = 10) -> Tensor:
    """Perform linear interpolation on a sequence with only one keyframe gap.

    This function accepts a tensor and the number front and back keyframes.
    The keyframes are preserved as is and the gap in the middle is filled
    with linear interpolation values.

    Args:
        x (Tensor): Input Tensor to interpolate. [..., N, ...].
                    N = front + keyframe_gap + back
        dim (int): Dimension to index
        front (int, optional): Length of initial keyframes. Defaults to 10.
        keyframe_gap (int, optional): Length of keyframe gap. Defaults to 30.
        back (int, optional): Length of final keyframes. Defaults to 10.
    
    Returns:
        Tensor: Linear Interpolated Tensor
    """
    # Define index tensor
    index_tensor = torch.arange(x.shape[dim]).unsqueeze(dim=1).to(x.device)

    # Extract keyframe boundaries
    first = x.index_select(dim, index_tensor[front - 1])
    last = x.index_select(dim, index_tensor[front + keyframe_gap])

    # Interpolate
    tile_dimension = [1 for i in range(len(x.shape))]
    tile_dimension[dim] = keyframe_gap + 2
    tile_dimension = tuple(tile_dimension)

    trans_fractions = torch.linspace(0, 1, keyframe_gap + 2).reshape(tile_dimension)
    offset = last - first

    trans_sequence = torch.tile(first, tile_dimension)
    trans_sequence = trans_sequence + trans_fractions * offset

    # Concatenate
    x = torch.cat([
        x.index_select(dim, index_tensor[:front - 1].squeeze()),
        trans_sequence,
        x.index_select(dim, index_tensor[front + keyframe_gap + 1:].squeeze())
    ], dim)

    return x

if __name__ == '__main__':
    x = Tensor([[[[1, 2]], [[2, 4]], [[3, 6]], [[4, 8]], [[5, 10]], [[6, 12]]]])
    fixed_points = LongTensor([0, 1, 4, 5])

    out = linear_interpolation(x, 1, fixed_points)
    out_ = single_linear_interpolation(x, 1, 2, 2, 2)

    print(out)

    print(out_)

    print(x.shape, out.shape, out_.shape)