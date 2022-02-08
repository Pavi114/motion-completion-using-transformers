from torch.nn import Module
from torch.nn.functional import mse_loss

class L2PLoss(Module):
    """nn.Module that calculates L2P Loss
    """

    def __init__(self) -> None:
        super(L2PLoss, self).__init__()

    def forward(self, x, x_cap):
        """Calculate L2P Loss

        Args:
            x: TODO based on model output
            x_cap: TODO based on transformer output
        """
        
        # Get Absolute Position
        p = x
        p_cap = x_cap

        # Calculate Loss
        return mse_loss(p, p_cap)

class L2QLoss(Module):
    """nn.Module that calculates L2Q Loss
    """

    def __init__(self) -> None:
        super(L2QLoss, self).__init__()

    def forward(self, x, x_cap):
        """Calculate L2Q Loss

        Args:
            x: TODO based on model output
            x_cap: TODO based on transformer output
        """
        
        # Get Global Quaternions
        q = x
        q_cap = x_cap

        # Calculate Loss
        return mse_loss(q, q_cap)