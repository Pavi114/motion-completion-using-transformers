import torch
from torch import Tensor

def round_tensor(x: Tensor, decimals: int = 0) -> Tensor:
    """Rounds the given tensor to `decimals` decimal places

    Args:
        x (Tensor): Tensor to round.
        decimals (int): Precision (Default = 0).

    Returns:
        Tensor: Rounded tensor
    """
    return torch.round(x * (10 ** decimals)) / 10 ** decimals