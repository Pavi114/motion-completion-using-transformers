import torch
from torch import Tensor

def moving_average_smoothing(x: Tensor, dim: int = -1, window_size: int = 1) -> Tensor:
    """Applies moving average smoothing to the given tensor.
    
    Args:
        x (Tensor): Tensor to smoothen.
        dim (int): Dimension to average.
        window_size (int): Window of moving average.

    Returns:
        Tensor: Smoothened tensor
    """
    index_tensor = torch.arange(x.shape[dim]).to(x.device)

    x_index = [torch.index_select(x, dim, index_tensor[i]) for i in range(x.shape[dim])]

    averaged_tensors = []

    for i in range(len(x_index)):
        n = 0
        s = 0
        for j in range(-window_size, window_size + 1):
            if i + j >= 0 and i + j < len(x_index):
                n += 1
                s += x_index[i + j]
        
        averaged_tensors.append(s / n)
    
    return torch.cat(averaged_tensors, dim=dim)    