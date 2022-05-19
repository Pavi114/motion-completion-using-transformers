import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Embedding, functional as F

class KeyframeEncoding(Module):
    """nn.Module that performs keyframe encoding
    """

    def __init__(self, d_model: int, device: torch.device = torch.device('cpu')) -> None:
        """
        Args:
            d_model (int): Embedding dimension. Must be even.
            max_len (int, optional): Maximum sequence length. Defaults to 5000.
        """
        super(KeyframeEncoding, self).__init__()

        self.device = device

        # Keyframe = 0, Unknown = 1, Ignored = 2
        self.embedding = Embedding(3, d_model)

        # Index tensor
        index_tensor = torch.arange(start=0, end=3, step=1).unsqueeze(dim=-1).to(device)
        self.register_buffer('index_tensor', index_tensor, persistent=False)

    def forward(self, x: Tensor, seq_len: int = 65, front: int = 10, back: int = 10, keyframe_gap: int = 30) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        front_keyframes = torch.tile(self.embedding(self.index_tensor[0]), (front, 1))
        back_keyframes = torch.tile(self.embedding(self.index_tensor[0]), (back, 1))
        unknown_frames = torch.tile(self.embedding(self.index_tensor[1]), (keyframe_gap, 1))
        ignored_frames = torch.tile(self.embedding(self.index_tensor[2]), (seq_len - front - back - keyframe_gap, 1))

        keyframe_embedding = torch.cat([front_keyframes, unknown_frames, back_keyframes, ignored_frames])

        return x + keyframe_embedding

if __name__ == '__main__':
    d_model = 4
    seq_len = 6

    model = KeyframeEncoding(d_model)

    x = torch.zeros(1, seq_len, d_model)

    y = model(x, 8, 2, 2, 2)

    print(x, x.shape)
    print(y, y.shape)