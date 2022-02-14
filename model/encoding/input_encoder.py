import torch
from torch import nn, Tensor
from constants import NUM_JOINTS, P_EMBEDDING_DIM, Q_EMBEDDING_DIM, V_EMBEDDING_DIM

from model.encoding.linear_encoding import LinearEncoding

class InputEncoder(nn.Module):
    """Encodes the input sequence.
    """

    def __init__(self) -> None:
        super(InputEncoder, self).__init__()

        self.q_encoder = LinearEncoding(input_size=4, hidden_size=16, output_size=Q_EMBEDDING_DIM)
        self.p_encoder = LinearEncoding(input_size=3, hidden_size=8, output_size=P_EMBEDDING_DIM)
        self.v_encoder = LinearEncoding(input_size=3, hidden_size=8, output_size=V_EMBEDDING_DIM)

    def forward(self, local_q: Tensor, root_p: Tensor, root_v: Tensor) -> Tensor:
        """
        Args:
            local_q (Tensor): Local quaternions. [batch_size, seq_len, J, 4]
            root_p (Tensor): Global Root Position. [batch_size, seq_len, 3]
            root_v (Tensor): Global Root Velocity. [batch_size, seq_len, 3]

        Returns:
            Tensor: Encoded Input
        """
        local_q = self.q_encoder(local_q)
        root_p = self.p_encoder(root_p)
        root_v = self.v_encoder(root_v)

        # Reshape Q
        local_q = local_q.reshape((local_q.shape[0], local_q.shape[1], NUM_JOINTS * Q_EMBEDDING_DIM))

        # Concateneate tensors
        x = torch.cat([root_p, root_v, local_q], dim=-1)

        return x