import argparse
import json
from pathlib import Path
from constants import DEVICE, MODEL_SAVE_DIRECTORY, OUTPUT_DIRECTORY, PARENTS

import torch
from torch.nn import functional as F
from model.encoding.input_encoder import InputEncoder
from model.encoding.output_decoder import OutputDecoder
from util.interpolation.fixed_points import get_fixed_points
from util.interpolation.interpolation_factory import get_p_interpolation, get_q_interpolation
from util.load_data import load_viz_dataset
from util.math import round_tensor
from util.quaternions import quat_fk
from model.transformer import Transformer
from util.read_config import read_config
from util.smoothing.moving_average_smoothing import moving_average_smoothing


def visualize(model_name='default'):
    # Load config
    config = read_config(model_name)

    # Load and Preprocess Data
    test_dataloader = load_viz_dataset(config['dataset'])

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

    transformer.eval()
    input_encoder.eval()
    output_decoder.eval()

    # Visualize
    viz_batch = next(iter(test_dataloader))

    local_q = round_tensor(viz_batch["local_q"].to(DEVICE), decimals=4)
    local_p = round_tensor(viz_batch["local_p"].to(DEVICE), decimals=4)
    root_p = round_tensor(viz_batch["X"][:, :, 0, :].to(DEVICE), decimals=4)
    root_v = round_tensor(viz_batch["root_v"].to(DEVICE), decimals=4)

    in_local_q = q_interpolation_function(local_q, 1, fixed_points)
    in_local_p = p_interpolation_function(local_p, 1, fixed_points)
    in_root_p = p_interpolation_function(root_p, 1, fixed_points)
    in_root_v = p_interpolation_function(root_v, 1, fixed_points)

    seq = input_encoder(in_local_q, in_root_p, in_root_v)

    out = transformer(seq, seq)

    ma_out = moving_average_smoothing(out, dim=1)

    out_q, out_p, out_v = output_decoder(out)

    ma_out_q, ma_out_p, ma_out_v = output_decoder(ma_out)

    out_local_p = local_p
    out_local_p[:, :, 0, :] = out_p

    ma_out_local_p = local_p
    ma_out_local_p[:, :, 0, :] = ma_out_p

    _, x = quat_fk(local_q.detach().cpu().numpy(),
                   local_p.detach().cpu().numpy(), PARENTS)
    _, in_x = quat_fk(in_local_q.detach().cpu().numpy(),
                      in_local_p.detach().cpu().numpy(), PARENTS)
    _, out_x = quat_fk(out_q.detach().cpu().numpy(),
                       out_local_p.detach().cpu().numpy(), PARENTS)
    _, ma_out_x = quat_fk(ma_out_q.detach().cpu().numpy(),
                       ma_out_local_p.detach().cpu().numpy(), PARENTS)

    for i in range(config['dataset']['batch_size']):
        output_dir = f'{OUTPUT_DIRECTORY}/viz/{model_name}/{i}'

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(f'{output_dir}/ground_truth.json', 'w') as f:
            f.truncate(0)
            f.write(json.dumps(x[i, :, :, :].tolist()))

        with open(f'{output_dir}/input.json', 'w') as f:
            f.truncate(0)
            f.write(json.dumps(in_x[i, :, :, :].tolist()))

        with open(f'{output_dir}/output.json', 'w') as f:
            f.truncate(0)
            f.write(json.dumps(out_x[i, :, :, :].tolist()))

        with open(f'{output_dir}/output_smoothened.json', 'w') as f:
            f.truncate(0)
            f.write(json.dumps(ma_out_x[i, :, :, :].tolist()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        help='Name of the model. Used for loading and saving weights.',
        type=str,
        default='default')

    args = parser.parse_args()

    visualize(args.model_name)
