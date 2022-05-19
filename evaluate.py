import argparse
from cgi import test
from pathlib import Path

import torch
from torch.nn import L1Loss
from tqdm import tqdm

from constants import DEVICE, MODEL_SAVE_DIRECTORY, OUTPUT_DIRECTORY, PARENTS
from model.encoding.input_encoder import InputEncoder
from model.encoding.output_decoder import OutputDecoder
from model.loss.fk_loss import FKLoss
from model.loss.l2_loss import L2PLoss, L2QLoss
from model.loss.npss_loss import NPSSLoss
from train_stats import load_stats
from util.interpolation.fixed_points import get_fixed_points
from util.interpolation.interpolation_factory import get_p_interpolation, get_q_interpolation
from util.interpolation.linear_interpolation import single_linear_interpolation
from util.interpolation.spherical_interpolation import single_spherical_interpolation
from util.load_data import load_test_dataset
from model.transformer import Transformer
from util.math import round_tensor
from util.read_config import read_config
from util.smoothing.moving_average_smoothing import moving_average_smoothing


def evaluate(model_name='default', keyframe_gap=30):
    # Load config
    config = read_config(model_name)

    # Load and Preprocess Data
    test_dataloader = load_test_dataset(config['dataset'])

    # Training Loop
    transformer = Transformer(config).to(DEVICE)

    input_encoder = InputEncoder(config['embedding_size']).to(DEVICE)

    output_decoder = OutputDecoder(config['embedding_size']).to(DEVICE)

    checkpoint = torch.load(
        f'{MODEL_SAVE_DIRECTORY}/model_{model_name}.pt', map_location=DEVICE)

    transformer.load_state_dict(checkpoint['transformer_state_dict'])
    input_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    output_decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Initialize losses
    l1_criterion = L1Loss()
    fk_criterion = FKLoss()
    l2p_criterion = L2PLoss()
    l2q_criterion = L2QLoss()
    npss_criterion = NPSSLoss()

    transformer.eval()
    input_encoder.eval()
    output_decoder.eval()

    global_q_loss = 0
    global_fk_loss = 0
    global_l2p_loss = 0
    global_l2q_loss = 0
    global_npss_loss = 0

    global_interpolation_q_loss = 0
    global_interpolation_fk_loss = 0
    global_interpolation_l2p_loss = 0
    global_interpolation_l2q_loss = 0
    global_interpolation_npss_loss = 0

    _, _, offsets, _ = load_stats()
    offsets = torch.Tensor(offsets).to(DEVICE)
    offsets = offsets.repeat((config['dataset']['batch_size'], config['dataset']['max_window_size'], 1, 1))

    # Visualize
    tqdm_dataloader = tqdm(test_dataloader)
    for index, batch in enumerate(tqdm_dataloader):
        local_q = round_tensor(batch["local_q"].to(DEVICE), decimals=4)
        local_p = round_tensor(batch["local_p"].to(DEVICE), decimals=4)
        root_p = round_tensor(batch["X"][:, :, 0, :].to(DEVICE), decimals=4)
        root_v = round_tensor(batch["root_v"].to(DEVICE), decimals=4)

        in_local_q = single_spherical_interpolation(
            local_q, dim=1, front=config['dataset']['front_pad'], keyframe_gap=keyframe_gap, back=config['dataset']['back_pad'])

        in_root_p = single_linear_interpolation(
            root_p, dim=1, front=config['dataset']['front_pad'], keyframe_gap=keyframe_gap, back=config['dataset']['back_pad'])
 
        in_root_v = single_linear_interpolation(
            root_v, dim=1, front=config['dataset']['front_pad'], keyframe_gap=keyframe_gap, back=config['dataset']['back_pad'])

        in_local_p = torch.cat([
            in_root_p.unsqueeze(dim=2),
            offsets
        ], dim=2)


        seq = input_encoder(in_local_q, in_root_p, in_root_v)

        out = transformer(seq, seq, keyframe_gap)

        out_q, out_p, out_v = output_decoder(out)

        out_q = out_q / torch.norm(out_q, dim=-1, keepdim=True)

        out_local_p = torch.cat([
            out_p.unsqueeze(dim=2),
            offsets
        ], dim=2)

        local_q = local_q[:, config['dataset']['front_pad']:config['dataset']['front_pad'] + keyframe_gap, ...]
        local_p = local_p[:, config['dataset']['front_pad']:config['dataset']['front_pad'] + keyframe_gap, ...]
        root_p = root_p[:, config['dataset']['front_pad']:config['dataset']['front_pad'] + keyframe_gap, ...]
        root_v = root_v[:, config['dataset']['front_pad']:config['dataset']['front_pad'] + keyframe_gap, ...]

        in_local_q = in_local_q[:, config['dataset']['front_pad']:config['dataset']['front_pad'] + keyframe_gap, ...]
        in_local_p = in_local_p[:, config['dataset']['front_pad']:config['dataset']['front_pad'] + keyframe_gap, ...]
        in_root_p = in_root_p[:, config['dataset']['front_pad']:config['dataset']['front_pad'] + keyframe_gap, ...]
        in_root_v = in_root_v[:, config['dataset']['front_pad']:config['dataset']['front_pad'] + keyframe_gap, ...]

        out_q = out_q[:, config['dataset']['front_pad']:config['dataset']['front_pad'] + keyframe_gap, ...]
        out_local_p = out_local_p[:, config['dataset']['front_pad']:config['dataset']['front_pad'] + keyframe_gap, ...]
        out_p = out_p[:, config['dataset']['front_pad']:config['dataset']['front_pad'] + keyframe_gap, ...]
        out_v = out_v[:, config['dataset']['front_pad']:config['dataset']['front_pad'] + keyframe_gap, ...]

        # Evaluate
        q_loss = l1_criterion(local_q, out_q).item()
        fk_loss = fk_criterion(local_p, local_q, out_local_p, out_q).item()
        l2p_loss = l2p_criterion(local_p, local_q, out_local_p, out_q).item()
        l2q_loss = l2q_criterion(local_p, local_q, out_local_p, out_q).item()
        npss_loss = npss_criterion(local_p, local_q, out_local_p, out_q).item()

        in_q_loss = l1_criterion(local_q, in_local_q).item()
        in_fk_loss = fk_criterion(
            local_p, local_q, in_local_p, in_local_q).item()
        in_l2p_loss = l2p_criterion(
            local_p, local_q, in_local_p, in_local_q).item()
        in_l2q_loss = l2q_criterion(
            local_p, local_q, in_local_p, in_local_q).item()
        in_npss_loss = npss_criterion(
            local_p, local_q, in_local_p, in_local_q).item()

        tqdm_dataloader.set_description(
            f"batch: {index + 1} | q: {q_loss:.4f} fk: {fk_loss:.4f} l2p: {l2p_loss:.4f} l2q: {l2q_loss:.4f} npss: {npss_loss:.4f}"
        )

        global_q_loss += q_loss
        global_fk_loss += fk_loss
        global_l2p_loss += l2p_loss
        global_l2q_loss += l2q_loss
        global_npss_loss += npss_loss

        global_interpolation_q_loss += in_q_loss
        global_interpolation_fk_loss += in_fk_loss
        global_interpolation_l2p_loss += in_l2p_loss
        global_interpolation_l2q_loss += in_l2q_loss
        global_interpolation_npss_loss += in_npss_loss

    # Store results
    path = f'{OUTPUT_DIRECTORY}/metrics'

    Path(path).mkdir(parents=True, exist_ok=True)

    s = f'Q: {global_q_loss / (index + 1)}\n' + \
        f'FK: {global_fk_loss / (index + 1)}\n' + \
        f'L2P: {global_l2p_loss / (index + 1)}\n' + \
        f'L2Q: {global_l2q_loss / (index + 1)}\n' + \
        f'NPSS: {global_npss_loss / (index + 1)}'

    in_s = f'IN_Q: {global_interpolation_q_loss / (index + 1)}\n' + \
        f'IN_FK: {global_interpolation_fk_loss / (index + 1)}\n' + \
        f'IN_L2P: {global_interpolation_l2p_loss / (index + 1)}\n' + \
        f'IN_L2Q: {global_interpolation_l2q_loss / (index + 1)}\n' + \
        f'IN_NPSS: {global_interpolation_npss_loss / (index + 1)}\n'

    with open(f'{path}/{model_name}.txt', 'w') as f:
        f.truncate(0)
        f.write(s)

    print(model_name, "\n", s, "\n", in_s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        help='Name of the model. Used for loading and saving weights.',
        type=str,
        default='default')

    parser.add_argument(
        '--keyframe_gap',
        help='Keyframe Gap for Visualization',
        type=int,
        default=30)

    args = parser.parse_args()

    evaluate(args.model_name, args.keyframe_gap)
