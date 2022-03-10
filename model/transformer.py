import torch

from torch import nn
from torch.nn import Module
from constants import DEVICE, NUM_JOINTS

from .encoding.positional_encoding import PositionalEncoding


class Transformer(Module):
    """nn.Module for transformer"""

    # Constructor
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dim_model = NUM_JOINTS * config['embedding_size']['q'] + config['embedding_size']['q'] + config['embedding_size']['v']

        self.positional_encoder = PositionalEncoding(
            d_model=self.dim_model,
            max_len=self.config['dataset']['window_size'],
            device=DEVICE)

        self.transformer = nn.Transformer(
            d_model=self.dim_model,
            nhead=self.config['model']['num_heads'],
            num_encoder_layers=self.config['model']['num_encoder_layers'],
            num_decoder_layers=self.config['model']['num_decoder_layers'],
            dropout=self.config['model']['dropout_p'],
            batch_first=True)

        self.register_buffer(
            'mask',
            self.get_target_mask(self.config['dataset']['window_size']).to(DEVICE))

    def forward(self, src, target):

        src = self.positional_encoder(src)
        target = self.positional_encoder(target)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        return self.transformer(
            src,
            target,
            # src_mask=self.mask,
            tgt_mask=self.mask)

    def get_target_mask(self, size) -> torch.Tensor:

        mask = torch.full((1, size), float("-inf"))
        # mask[0, 0] = 0
        # mask[0, 1] = 0
        # mask[0, 2] = 0
        # mask[0, -3] = 0
        # mask[0, -2] = 0
        # mask[0, -1] = 0

        # Unmask every KEYFRAME_GAP frame
        mask[0, ::self.config['dataset']['keyframe_gap']] = 0

        mask = mask.repeat(size, 1)

        return mask
