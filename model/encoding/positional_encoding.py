import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

class PositionalEncoding(Module):
    """nn.Module that performs positional encoding
    """

    def __init__(self, d_model: int, max_len: int = 5000, device: torch.device = torch.device('cpu')) -> None:
        """
        Args:
            d_model (int): Embedding dimension. Must be even.
            max_len (int, optional): Maximum sequence length. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()

        # Tensor[max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)

        # Tensor[1, d_model / 2]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        # Tensor[max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # Set all even terms to Tensor[max_len,  d_model / 2]
        pe[:, 0::2] = torch.sin(position * div_term)

        # Set all odd terms to Tensor[max_len, d_model / 2]
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.to(device)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(1)]

if __name__ == '__main__':
    d_model = 256
    seq_len = 128

    model = PositionalEncoding(d_model, seq_len)
    x = torch.zeros(seq_len, 1, d_model)

    y = model(x).squeeze()

    plt.imshow(y)
    plt.show()