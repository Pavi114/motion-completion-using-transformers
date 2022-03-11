import numpy as np

from torch.nn import Module

from model.loss.l2_loss import L2QLoss


class NPSSLoss(Module):
    """nn.Module that calculates NPSS Loss
    
    Based on: https://arxiv.org/abs/1809.03036
    """

    def __init__(self) -> None:
        super(L2QLoss, self).__init__()

    def forward(self, x, x_cap):
        """Computes NPSS loss between the ground truth and 
        the generated animation

        Args:
            x (Tensor): Ground truth value
            x_cap (Tensor): Generated value

        Returns:
            float: npss loss 
        """
        
        # compute fourier coefficients 
        x_ftt_coeff = np.fft.fft(x, axis=-1)
        x_cap_fft_coeff = np.fft.fft(x_cap, axis=-1)

        #Sq the coeff
        x_ftt_coeff_sq = np.square(x_ftt_coeff)
        x_cap_ftt_coeff_sq = np.square(x_cap_fft_coeff)

        # sum the tensor
        x_tot = np.sum(x_ftt_coeff_sq, axis=-1)
        x_cap_tot = np.sum(x_cap_ftt_coeff_sq, axis=-1)

        # normalize
        x_norm = x_ftt_coeff_sq / x_tot
        x_cap_norm = x_cap_fft_coeff / x_cap_tot

        # Compute emd
        emd = np.linalg.norm((np.cumsum(x_cap_norm, axis=1) - np.cumsum(x_norm, axis=1)), ord=1, axis=1)

        # Weighted Avg (NPSS)
        npss_loss = np.average(emd, weights=x_tot)

        return npss_loss