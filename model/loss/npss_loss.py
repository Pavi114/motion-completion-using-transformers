from torch.nn import Module

from model.loss.l2_loss import L2QLoss

class NPSSLoss(Module):
    """nn.Module that calculates NPSS Loss
    
    Based on: https://arxiv.org/abs/1809.03036
    """

    def __init__(self) -> None:
        super(L2QLoss, self).__init__()

    def forward(self, x, x_cap):
        pass