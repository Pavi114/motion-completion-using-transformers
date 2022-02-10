import torch
from torch.nn import Module
from torch.nn.functional import l1_loss

class L1Loss(Module):
    """nn.Module that calculates L1 Loss (position + quaternion)
    """
    def __init__(self) -> None:
        super(L1Loss, self).__init__()

    def forward(self, x, x_cap):
        """Calculate L1 Loss

        Args:
            x: TODO based on model output
            x_cap: TODO based on transformer output
        """

        # Get absolute positions
        p = x
        p_cap = x_cap

        # Get global quaternions
        # q = x
        # q_cap = x_cap

        # Calculate Loss
        return l1_loss(p, p_cap) #+ l1_loss(q, q_cap)

if __name__ == '__main__':
    x = torch.rand((2, 3))
    x_cap = torch.rand((2, 3))

    l1 = L1Loss()

    print(x, x_cap, l1(x, x_cap))

