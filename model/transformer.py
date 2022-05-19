import torch

from torch import nn
from torch.nn import Module
from constants import DEVICE, NUM_JOINTS
from model.encoding.keyframe_encoding import KeyframeEncoding

from .encoding.positional_encoding import PositionalEncoding


class Transformer(Module):
    """nn.Module for transformer"""

    # Constructor
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dim_model = NUM_JOINTS * config['embedding_size']['q'] + \
            config['embedding_size']['q'] + config['embedding_size']['v']
        self.max_len = self.config['dataset']['max_window_size']
        self.front_pad = self.config['dataset']['front_pad']
        self.back_pad = self.config['dataset']['back_pad']

        self.positional_encoder = PositionalEncoding(
            d_model=self.dim_model,
            max_len=self.config['dataset']['max_window_size'],
            device=DEVICE)

        self.keyframe_encoder = KeyframeEncoding(
            d_model=self.dim_model,
            device=DEVICE
        )

        self.transformer = nn.Transformer(
            d_model=self.dim_model,
            nhead=self.config['model']['num_heads'],
            num_encoder_layers=self.config['model']['num_encoder_layers'],
            num_decoder_layers=self.config['model']['num_decoder_layers'],
            dropout=self.config['model']['dropout_p'],
            batch_first=True)

    def forward(self, src, target, keyframe_gap):
        src = self.positional_encoder(src)
        src = self.keyframe_encoder(
            src, self.max_len, self.front_pad, self.back_pad, keyframe_gap)

        target = self.positional_encoder(target)
        target = self.keyframe_encoder(
            target, self.max_len, self.front_pad, self.back_pad, keyframe_gap)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        return self.transformer(
            src,
            target,
            # src_mask=self.mask,
            tgt_mask=self.get_target_mask(keyframe_gap))

    def get_target_mask(self, keyframe_gap) -> torch.Tensor:
        front_mask = torch.zeros((1, self.front_pad)).to(DEVICE)
        back_mask = torch.zeros((1, self.back_pad)).to(DEVICE)
        trans_mask = torch.full((1, keyframe_gap), float("-inf")).to(DEVICE)
        ignore_mask = torch.full(
            (1, self.max_len - (self.front_pad + self.back_pad + keyframe_gap)), float("-inf")).to(DEVICE)

        mask = torch.cat([
            front_mask,
            trans_mask,
            back_mask,
            ignore_mask
        ], dim=1)

        mask = mask.repeat(self.max_len, 1)

        return mask
