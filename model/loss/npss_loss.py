import numpy as np
import torch

from torch.nn import Module

class NPSSLoss(Module):
    """nn.Module that calculates NPSS Loss
    
    Based on: https://arxiv.org/abs/1809.03036
    """

    def __init__(self) -> None:
        super(NPSSLoss, self).__init__()

    def forward(self, x, x_cap):
        """Computes NPSS loss between the ground truth and 
        the generated animation

        Args:
            x (Tensor): Ground truth value
            x_cap (Tensor): Generated value

        Returns:
            float: npss loss 
        """

        # Reshape to have all features in one dimension
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x_cap = x_cap.reshape((x_cap.shape[0], x_cap.shape[1], x_cap.shape[2] * x_cap.shape[3]))
        
        # compute fourier coefficients 
        x_ftt_coeff = torch.fft.fft(x, axis=-1)
        x_cap_fft_coeff = torch.fft.fft(x_cap, axis=-1)

        #Sq the coeff
        x_ftt_coeff_sq = torch.square(x_ftt_coeff)
        x_cap_ftt_coeff_sq = torch.square(x_cap_fft_coeff)

        # sum the tensor
        x_tot = torch.sum(x_ftt_coeff_sq, axis=-2).unsqueeze(dim=-2)
        x_cap_tot = torch.sum(x_cap_ftt_coeff_sq, axis=-2).unsqueeze(dim=-2)

        # normalize
        x_norm = x_ftt_coeff_sq / x_tot
        x_cap_norm = x_cap_fft_coeff / x_cap_tot

        # Compute emd
        emd = torch.linalg.norm((torch.cumsum(x_cap_norm, axis=1) - torch.cumsum(x_norm, axis=1)), ord=1, axis=1)

        # Find total norm
        x_norm_tot = torch.sum(x_norm, axis=-2)

        # Weighted Avg (NPSS)
        npss_loss = torch.mean(torch.real(emd * x_norm_tot))

        return npss_loss