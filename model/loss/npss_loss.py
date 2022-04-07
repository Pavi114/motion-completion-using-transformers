import numpy as np

import torch
from torch import Tensor
from torch.nn import Module

from constants import PARENTS
from util.quaternions import quat_fk_tensor

class NPSSLoss(Module):
    """nn.Module that calculates NPSS Loss
    
    Based on: https://arxiv.org/abs/1809.03036
    """

    def __init__(self) -> None:
        super(NPSSLoss, self).__init__()

    def forward(self, local_p: Tensor, local_q: Tensor, local_p_cap: Tensor, local_q_cap: Tensor) -> Tensor:
        """
        Args:
            local_p (Tensor): Local positions [..., J, 3]
            local_q (Tensor): Local quaternions [..., J, 4]
            local_p_cap (Tensor): Predicted Local positions [..., J, 3]
            local_q_cap (Tensor): Predicted Local quaternions [..., J, 4]

        Returns:
            Tensor: NPSS Loss.
        """

        # Get global quaternions
        q, _ = quat_fk_tensor(local_q, local_p, PARENTS)
        q_cap, _ = quat_fk_tensor(local_q_cap, local_p_cap, PARENTS)

        x = q
        x_cap = q_cap

        # Reshape to have all features in one dimension
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x_cap = x_cap.reshape((x_cap.shape[0], x_cap.shape[1], x_cap.shape[2] * x_cap.shape[3]))

        # compute fourier coefficients 
        x_ftt_coeff = torch.real(torch.fft.fft(x, axis=1))
        x_cap_fft_coeff = torch.real(torch.fft.fft(x_cap, axis=1))

        #Sq the coeff
        x_ftt_coeff_sq = torch.square(x_ftt_coeff)
        x_cap_ftt_coeff_sq = torch.square(x_cap_fft_coeff)

        # sum the tensor
        x_tot = torch.sum(x_ftt_coeff_sq, axis=1, keepdim=True)
        x_cap_tot = torch.sum(x_cap_ftt_coeff_sq, axis=1, keepdim=True)

        # normalize
        x_norm = x_ftt_coeff_sq / x_tot
        x_cap_norm = x_cap_ftt_coeff_sq / x_cap_tot

        # Compute emd
        emd = torch.norm(x_norm - x_cap_norm, dim=1, p=1)

        # Find total norm
        x_norm_tot = torch.sum(x_norm, axis=1)

        # Weighted Avg (NPSS)
        npss_loss = torch.sum(emd * x_norm_tot) / torch.sum(x_norm_tot)

        return npss_loss