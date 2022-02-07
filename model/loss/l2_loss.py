from torch.nn import Module

class L2PLoss(Module):
    """nn.Module that calculates L2P Loss
    """

    def __init__(self) -> None:
        super(L2PLoss, self).__init__()

    def forward(self, x, x_cap):
        pass

class L2QLoss(Module):
    """nn.Module that calculates L2Q Loss
    """

    def __init__(self) -> None:
        super(L2QLoss, self).__init__()

    def forward(self, x, x_cap):
        pass