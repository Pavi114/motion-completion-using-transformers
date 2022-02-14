from torch import nn, Tensor

class LinearEncoding(nn.Module):
    """nn.Module that performs linear encoding.
    Specify [input_size, hidden_size, output_size] for a 2 layer NN
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(LinearEncoding, self).__init__()

        self.l1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.l2 = nn.Linear(in_features=hidden_size, out_features=output_size)

        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        return x
