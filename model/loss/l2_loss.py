from torch import Tensor
from torch.nn import Module
from torch.nn.functional import mse_loss
from constants import PARENTS

from util.quaternions import quat_fk_tensor

class L2PLoss(Module):
    """nn.Module that calculates L2P Loss
    """

    def __init__(self) -> None:
        super(L2PLoss, self).__init__()

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

        # Calculate Loss
        return mse_loss(q, q_cap)