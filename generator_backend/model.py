
import torch
from constants import DEVICE, MODEL_SAVE_DIRECTORY, PARENTS
from model.encoding.input_encoder import InputEncoder
from model.encoding.output_decoder import OutputDecoder
from model.transformer import Transformer
from util.interpolation.fixed_points import get_fixed_points
from util.interpolation.interpolation_factory import get_p_interpolation, get_q_interpolation
from util.quaternions import quat_fk, quat_ik_tensor
from util.read_config import read_config

class Model:
    def __init__(self, model_name) -> None:
        # Load config
        self.config = read_config(model_name)

        self.transformer = Transformer(self.config).to(DEVICE)

        self.input_encoder = InputEncoder(self.config['embedding_size']).to(DEVICE)

        self.output_decoder = OutputDecoder(self.config['embedding_size']).to(DEVICE)

        self.fixed_points = get_fixed_points(self.config['dataset']['window_size'], self.config['dataset']['keyframe_gap'])

        self.p_interpolation_function = get_p_interpolation(self.config['hyperparameters']['interpolation'])
        self.q_interpolation_function = get_q_interpolation(self.config['hyperparameters']['interpolation'])

        checkpoint = torch.load(f'{MODEL_SAVE_DIRECTORY}/model_{model_name}.pt')

        self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        self.input_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.output_decoder.load_state_dict(checkpoint['decoder_state_dict'])

    def generate(self, gpos):
        gpos = torch.tensor(gpos).to(DEVICE)

        z_gpos = torch.zeros((128, 22, 3)).to(DEVICE)

        for i in range(127):
            z_gpos[i] = gpos[30 * (i // 30)]

        z_gpos[-1] = gpos[-1]

        in_gpos = self.p_interpolation_function(gpos, 0, self.fixed_points)

        return z_gpos.cpu().detach().numpy().tolist(), in_gpos.cpu().detach().numpy().tolist()

    def _generate(self, gpos):
        grot = torch.tensor(grot).to(DEVICE)
        gpos = torch.tensor(gpos).to(DEVICE)

        lrot, lpos = quat_ik_tensor(grot, gpos, PARENTS)

        lrotSeq = torch.zeros((128, 22, 4)).to(DEVICE)
        lposSeq = torch.zeros((128, 22, 3)).to(DEVICE)
        rootPosSeq = torch.zeros((128, 3)).to(DEVICE)
        rootVSeq = torch.zeros((128, 3)).to(DEVICE)

        for i in range(5):
            lrotSeq[30*i] = lrot[i]
            lposSeq[30*i] = lpos[i]
            rootPosSeq[30*i] = lpos[i][0]

        lrotSeq[-1] = lrot[-1]
        rootPosSeq[-1] = lpos[-1][0]

        in_local_q = self.q_interpolation_function(lrotSeq, -3, self.fixed_points)
        in_local_p = self.p_interpolation_function(lposSeq, -3, self.fixed_points)
        in_root_p = self.p_interpolation_function(rootPosSeq, -2, self.fixed_points)
        in_root_v = self.p_interpolation_function(rootVSeq, -2, self.fixed_points)

        seq = self.input_encoder(in_local_q, in_root_p, in_root_v)

        out = self.transformer(seq, seq)

        out_q, out_p, out_v = self.output_decoder(out)

        out_q = out_q / torch.norm(out_q, dim=-1, keepdim=True)

        out_local_p = in_local_p
        out_local_p[:, :, 0, :] = out_p

        _, x = quat_fk(lposSeq.detach().cpu().numpy(),
                   lposSeq.detach().cpu().numpy(), PARENTS)
        _, in_x = quat_fk(in_local_q.detach().cpu().numpy(),
                        in_local_p.detach().cpu().numpy(), PARENTS)
        _, out_x = quat_fk(out_q.detach().cpu().numpy(),
                        out_local_p.detach().cpu().numpy(), PARENTS)

        return x, in_x, out_x
