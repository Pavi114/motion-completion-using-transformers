from typing import Tuple
import torch
from torch import nn, Tensor
from hyperparameters import NUM_JOINTS, P_EMBEDDING_DIM, Q_EMBEDDING_DIM, V_EMBEDDING_DIM

from model.encoding.linear_encoding import LinearEncoding

class OutputDecoder(nn.Module):
    """Encodes the input sequence.
    """

    def __init__(self) -> None:
        super(OutputDecoder, self).__init__()

        self.q_decoder = LinearEncoding(input_size=Q_EMBEDDING_DIM, hidden_size=16, output_size=4)
        self.p_decoder = LinearEncoding(input_size=P_EMBEDDING_DIM, hidden_size=8, output_size=3)
        self.v_decoder = LinearEncoding(input_size=V_EMBEDDING_DIM, hidden_size=8, output_size=3)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """[summary]

        Args:
            x (Tensor): Output Tensor. 
                [batch_size, seq_len, J * Q_EMBEDDING_DIM + P_EMBEDDING_DIM + V_EMBEDDING_DIM]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: [local_q, root_p, root_v]
        """
        # Extract three components
        root_p = x[:, :, :P_EMBEDDING_DIM]
        root_v = x[:, :, P_EMBEDDING_DIM:P_EMBEDDING_DIM + V_EMBEDDING_DIM]
        local_q = x[:, :, P_EMBEDDING_DIM + V_EMBEDDING_DIM:]

        # Reshape Q
        local_q = local_q.reshape((local_q.shape[0], local_q.shape[1], NUM_JOINTS, Q_EMBEDDING_DIM))

        # Decode Tensors
        local_q = self.q_decoder(local_q)
        root_p = self.p_decoder(root_p)
        root_v = self.v_decoder(root_v)

        return local_q, root_p, root_v