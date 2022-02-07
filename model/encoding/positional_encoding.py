from torch.nn import Module

class PositionalEncoding(Module):
    """nn.Module that performs positional encoding
    """

    def __init__(self) -> None:
        super(PositionalEncoding, self).__init__()

    def forward(self, x):
        pass