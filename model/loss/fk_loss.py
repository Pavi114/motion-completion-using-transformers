from torch import Tensor
from torch.nn import Module
from torch.nn.functional import l1_loss
from constants import PARENTS

from util.quaternions import quat_fk_tensor

class FKLoss(Module):
    """nn.Module that calculates forward kinematics loss
    """
    def __init__(self) -> None:
        super(FKLoss, self).__init__()

    def forward(self, local_p: Tensor, local_q: Tensor, local_p_cap: Tensor, local_q_cap: Tensor) -> Tensor:
        """
        Args:
            local_p (Tensor): Local positions [..., J, 3]
            local_q (Tensor): Local quaternions [..., J, 4]
            local_p_cap (Tensor): Predicted Local positions [..., J, 3]
            local_q_cap (Tensor): Predicted Local quaternions [..., J, 4]

        Returns:
            Tensor: FK Loss.
        """

        # Get globals
        q, x = quat_fk_tensor(local_q, local_p, PARENTS)

        q_cap, x_cap = quat_fk_tensor(local_q_cap, local_p_cap, PARENTS)

        # Calculate Loss
        return l1_loss(x, x_cap)