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

if __name__ == '__main__':
    x = Tensor([[[[1, 2]], [[0, 0]], [[3, 6]], [[0, 0]], [[5, 10]]]])
    fixed_points = LongTensor([0, 2, 4])

    out = linear_interpolation(x, 1, fixed_points)

    print(out)

    print(x.shape, out.shape)