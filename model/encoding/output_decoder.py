from typing import Tuple
from torch import nn, Tensor
from constants import NUM_JOINTS
from model.encoding.linear_encoding import LinearEncoding

class OutputDecoder(nn.Module):
    """Encodes the input sequence.
    """

    def __init__(self, embedding_size) -> None:
        super(OutputDecoder, self).__init__()

        self.embedding_size = embedding_size

        self.q_decoder = LinearEncoding(input_size=self.embedding_size['q'], hidden_size=16, output_size=4)
        self.p_decoder = LinearEncoding(input_size=self.embedding_size['p'], hidden_size=8, output_size=3)
        self.v_decoder = LinearEncoding(input_size=self.embedding_size['v'], hidden_size=8, output_size=3)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """[summary]

        Args:
            x (Tensor): Output Tensor. 
                [batch_size, seq_len, J * Q_EMBEDDING_DIM + P_EMBEDDING_DIM + V_EMBEDDING_DIM]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: [local_q, root_p, root_v]
        """
        # Extract three components
        root_p = x[:, :, :self.embedding_size['p']]
        root_v = x[:, :, self.embedding_size['p']:self.embedding_size['p'] + self.embedding_size['v']]
        local_q = x[:, :, self.embedding_size['p'] + self.embedding_size['v']:]

        # Reshape Q
        local_q = local_q.reshape((local_q.shape[0], local_q.shape[1], NUM_JOINTS, self.embedding_size['q']))

        # Decode Tensors
        local_q = self.q_decoder(local_q)
        root_p = self.p_decoder(root_p)
        root_v = self.v_decoder(root_v)

        return local_q, root_p, root_v