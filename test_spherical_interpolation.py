import torch

from util.interpolation.spherical_interpolation import spherical_interpolation

q = torch.Tensor([
    [
        [0, 0.408, 0.408, 0.816], 
        [1, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ]
])
fixed_points = torch.LongTensor([0, 3])

print(q.shape)

out = spherical_interpolation(q, -3, fixed_points)

print(q, q.shape)

print(out, out.shape)