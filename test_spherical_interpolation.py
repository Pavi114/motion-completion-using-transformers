import torch

from util.interpolation.spherical_interpolation import single_spherical_interpolation, spherical_interpolation

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

out = spherical_interpolation(q, -3, fixed_points)
out_ = single_spherical_interpolation(q, -3, 1, 2, 1)

print(out, out.shape)
print(out_, out_.shape)