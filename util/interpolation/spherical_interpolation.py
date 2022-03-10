import torch
from torch import LongTensor, Tensor

def _spherical_interpolation(x1: Tensor, x2: Tensor, alpha=0.2) -> Tensor:
    dot_pdt = torch.dot(x1, x2)
    i = min(max(dot_pdt, -1), 1)
    theta = torch.acos(torch.Tensor(i)) * alpha
    x3 = x2 - x1 * dot_pdt
    return x1 * torch.cos(theta) + x3 * torch.sin(theta)


def spherical_interpolation(x: Tensor, dim: int, fixed_points: LongTensor) -> Tensor:
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

    for i in range(len(fixed_points) - 1):

        print(fixed_values.index_select(dim, torch.LongTensor([i])).shape)
        _shape = fixed_values.index_select(dim, torch.LongTensor([i])).shape
        l = _spherical_interpolation(fixed_values.index_select(dim, torch.LongTensor([i])).reshape(-1), fixed_values.index_select(dim, torch.LongTensor([i + 1])).reshape(-1))
        # n = fixed_points[i + 1] - fixed_points[i]

        # delta = (fixed_values.index_select(dim, torch.LongTensor([i + 1])) - fixed_values.index_select(dim, torch.LongTensor([i]))) / n

        # d_range = []

        # # TODO: Opttorch.Tensor(imize
        # for j in range(n):
        #     d_range.append((fixed_values.index_select(dim, torch.LongTensor([i])) + delta * j))

        # xi.append(torch.cat(d_range, dim=dim))

        # # print(1, delta.shape)

        # # print(2, torch.arange(0, n).unsqueeze(dim=0), torch.arange(0, n).unsqueeze(dim=0).shape)

        # # d_range =  delta * torch.arange(0, n)
        # # print(3, d_range, d_range.shape)
        print(l.reshape(_shape))
        xi.append(fixed_values.index_select(dim, torch.LongTensor([i])))
        xi.append(l.reshape(_shape))

    xi.append(fixed_values.index_select(dim, torch.LongTensor([fixed_values.shape[dim] - 1])))

    return torch.cat(xi, dim=dim)

if __name__ == '__main__':
    x = Tensor([[[[1, 2]], [[0, 0]], [[3, 6]], [[0, 0]], [[5, 10]]]])
    fixed_points = LongTensor([0, 2, 4])

    out = spherical_interpolation(x, 1, fixed_points)

    print(out)

    print(x.shape, out.shape)