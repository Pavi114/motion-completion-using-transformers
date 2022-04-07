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
from util.interpolation.fixed_points import get_fixed_points
from util.interpolation.interpolation_factory import get_p_interpolation, get_q_interpolation
from util.load_data import load_test_dataset
from model.transformer import Transformer
from util.math import round_tensor
from util.read_config import read_config

def evaluate(model_name='default'):
    # Load config
    config = read_config(model_name)

    # Load and Preprocess Data
    test_dataloader = load_test_dataset(config['dataset'])

    # Training Loop
    transformer = Transformer(config).to(DEVICE)

    input_encoder = InputEncoder(config['embedding_size']).to(DEVICE)

    output_decoder = OutputDecoder(config['embedding_size']).to(DEVICE)

    fixed_points = get_fixed_points(config['dataset']['window_size'], config['dataset']['keyframe_gap'])

    p_interpolation_function = get_p_interpolation(config['hyperparameters']['interpolation'])
    q_interpolation_function = get_q_interpolation(config['hyperparameters']['interpolation'])

    checkpoint = torch.load(f'{MODEL_SAVE_DIRECTORY}/model_{model_name}.pt')

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

    # Visualize
    tqdm_dataloader = tqdm(test_dataloader)
    for index, batch in enumerate(tqdm_dataloader):
        local_q = round_tensor(batch["local_q"].to(DEVICE), decimals=4)
        local_p = round_tensor(batch["local_p"].to(DEVICE), decimals=4)
        root_p = round_tensor(batch["X"][:, :, 0, :].to(DEVICE), decimals=4)
        root_v = round_tensor(batch["root_v"].to(DEVICE), decimals=4)

        in_local_q = q_interpolation_function(local_q, 1, fixed_points)
        in_root_p = p_interpolation_function(root_p, 1, fixed_points)
        in_root_v = p_interpolation_function(root_v, 1, fixed_points)

        seq = input_encoder(in_local_q, in_root_p, in_root_v)

        out = transformer(seq, seq)

        out_q, out_p, out_v = output_decoder(out)

        out_q = out_q / torch.norm(out_q, dim=-1, keepdim=True)

        out_local_p = local_p
        out_local_p[:, :, 0, :] = out_p

        # Evaluate
        q_loss = l1_criterion(local_q, out_q).item()
        fk_loss = fk_criterion(local_p, local_q, out_local_p, out_q).item()
        l2p_loss = l2p_criterion(local_p, local_q, out_local_p, out_q).item()
        l2q_loss = l2q_criterion(local_p, local_q, out_local_p, out_q).item()
        npss_loss = npss_criterion(local_p, local_q, out_local_p, out_q).item()

        tqdm_dataloader.set_description(
            f"batch: {index + 1} | q: {q_loss:.4f} fk: {fk_loss:.4f} l2p: {l2p_loss:.4f} l2q: {l2q_loss:.4f} npss: {npss_loss:.4f}"
        )

        global_q_loss += q_loss
        global_fk_loss += fk_loss
        global_l2p_loss += l2p_loss
        global_l2q_loss += l2q_loss
        global_npss_loss += npss_loss

    # Store results
    path = f'{OUTPUT_DIRECTORY}/metrics'
    
    Path(path).mkdir(parents=True, exist_ok=True)

    with open(f'{path}/{model_name}.txt', 'w') as f:
        f.truncate(0)
        f.write(
            f'Q: {global_q_loss / index}\n' +
            f'FK: {global_fk_loss / index}\n' +
            f'L2P: {global_l2p_loss / index}\n' +
            f'L2Q: {global_l2q_loss / index}\n' +
            f'NPSS: {global_npss_loss / index}'
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        help='Name of the model. Used for loading and saving weights.',
        type=str,
        default='default')

    args = parser.parse_args()

    evaluate(args.model_name)
