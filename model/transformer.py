import torch

from torch import nn
from torch.nn import Module


from .encoding.positional_encoding import PositionalEncoding


class Transformer(Module):
    """nn.Module for transformer"""

    # Constructor
    def __init__(
        self,
        dim_model,
        num_heads,
        seq_len,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        self.positional_encoder = PositionalEncoding(d_model=dim_model, max_len=seq_len)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )
        self.register_buffer('mask', self.get_target_mask(seq_len))

    def forward(
        self, src, target
    ):

        src = self.positional_encoder(src)
        target = self.positional_encoder(target)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        return self.transformer(
            src,
            target,
            src_mask=self.mask,
            tgt_mask=self.mask
        )

    def get_target_mask(self, size) -> torch.Tensor:

        mask = torch.full((1, size), float("-inf"))
        mask[0, 0] = 0
        mask[0, -1] = 0
        mask = mask.repeat(size, 1)

        # [0, -inf, -inf ....., 0]
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask
