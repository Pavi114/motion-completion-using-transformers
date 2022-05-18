from torch import Tensor
import torch
from torch.nn import Module
from torch.nn.functional import mse_loss
from constants import DEVICE, PARENTS

from util.quaternions import quat_fk_tensor
from train_stats import load_stats

class L2PLoss(Module):
    """nn.Module that calculates L2P Loss
    """

    def __init__(self) -> None:
        super(L2PLoss, self).__init__()
        x_mean_np, x_std_np = load_stats()

        self.x_mean = Tensor(x_mean_np).to(DEVICE)
        self.x_std = Tensor(x_std_np).to(DEVICE)

    def forward(self, local_p: Tensor, local_q: Tensor, local_p_cap: Tensor, local_q_cap: Tensor) -> Tensor:
        """
        Args:
            local_p (Tensor): Local positions [..., J, 3]
            local_q (Tensor): Local quaternions [..., J, 4]
            local_p_cap (Tensor): Predicted Local positions [..., J, 3]
            local_q_cap (Tensor): Predicted Local quaternions [..., J, 4]

        Returns:
            Tensor: L2P Loss.
        """

        # Get globals
        _, x = quat_fk_tensor(local_q, local_p, PARENTS)

        _, x_cap = quat_fk_tensor(local_q_cap, local_p_cap, PARENTS)

        # Normalize
        x = (x - self.x_mean) / self.x_std
        x_cap = (x_cap - self.x_mean) / self.x_std

        return torch.mean(torch.sqrt(torch.sum((x - x_cap)**2, axis=-1)))

        # Calculate Loss
        return mse_loss(x, x_cap)

class L2QLoss(Module):
    """nn.Module that calculates L2Q Loss
    """

    def __init__(self) -> None:
        super(L2QLoss, self).__init__()

    def forward(self, local_p: Tensor, local_q: Tensor, local_p_cap: Tensor, local_q_cap: Tensor) -> Tensor:
        """
        Args:
            local_p (Tensor): Local positions [..., J, 3]
            local_q (Tensor): Local quaternions [..., J, 4]
            local_p_cap (Tensor): Predicted Local positions [..., J, 3]
            local_q_cap (Tensor): Predicted Local quaternions [..., J, 4]

        Returns:
            Tensor: L2P Loss.
        """

        # Get globals
        q, _ = quat_fk_tensor(local_q, local_p, PARENTS)

        q_cap, _ = quat_fk_tensor(local_q_cap, local_p_cap, PARENTS)

        # Normalize
        q = q / torch.norm(q, dim=-1, keepdim=True)
        q_cap = q_cap / torch.norm(q_cap, dim=-1, keepdim=True)

        return torch.mean(torch.sqrt(torch.sum((q - q_cap)**2, axis=-1)))

        # Calculate Loss
        return mse_loss(q, q_cap)