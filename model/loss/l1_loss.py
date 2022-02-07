from torch.nn import Module

class L1Loss(Module):
    """nn.Module that calculates L1 Loss (position + quaternion)
    """

    def __init__(self) -> None:
        super(L1Loss, self).__init__()

    def forward(self, x, x_cap):
        pass