import argparse
import json
from pathlib import Path
from constants import DEVICE, DIM_MODEL, FIXED_POINTS
from hyperparameters import *

import torch
from model.encoding.input_encoder import InputEncoder
from model.encoding.output_decoder import OutputDecoder
from util.interpolation.linear_interpolation import linear_interpolation
from util.load_data import load_test_dataset
from util.quaternions import quat_fk
from model.transformer import Transformer


def visualize(model_name='default'):
    # Load and Preprocess Data
    test_dataloader = load_test_dataset(LAFAN1_DIRECTORY)

    # Training Loop
    transformer = Transformer(
        dim_model=DIM_MODEL,
        num_heads=NUM_HEADS,
        seq_len=WINDOW_SIZE,
        num_encoder_layers=NUM_ENCODER,
        num_decoder_layers=NUM_DECODER,
        dropout_p=DROPOUT_P,
        device=DEVICE
    ).to(DEVICE)

    input_encoder = InputEncoder().to(DEVICE)

    output_decoder = OutputDecoder().to(DEVICE)

    checkpoint = torch.load(f'{MODEL_SAVE_DIRECTORY}/model_{model_name}.pt')

    transformer.load_state_dict(checkpoint['transformer_state_dict'])
    input_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    output_decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Visualize
    viz_batch = next(iter(test_dataloader))

    local_q = viz_batch["local_q"][:, :, :, :].to(DEVICE)
    local_p = viz_batch["local_p"][:, :, :, :].to(DEVICE)
    root_p = viz_batch["X"][:, :, 0, :].to(DEVICE)
    root_v = viz_batch["root_v"][:, :, :].to(DEVICE)

    in_local_q = linear_interpolation(local_q, 1, FIXED_POINTS)
    in_local_p = linear_interpolation(local_p, 1, FIXED_POINTS)
    in_root_p = linear_interpolation(root_p, 1, FIXED_POINTS)
    in_root_v = linear_interpolation(root_v, 1, FIXED_POINTS)

    seq = input_encoder(in_local_q, in_root_p, in_root_v)

    out = transformer(seq, seq)

    out_q, out_p, out_v = output_decoder(out)

    out_local_p = local_p
    out_local_p[:, :, 0, :] = out_p

    _, x = quat_fk(local_q.detach().cpu().numpy(),
                   local_p.detach().cpu().numpy(), PARENTS)
    _, in_x = quat_fk(in_local_q.detach().cpu().numpy(),
                      in_local_p.detach().cpu().numpy(), PARENTS)
    _, out_x = quat_fk(out_q.detach().cpu().numpy(),
                       out_local_p.detach().cpu().numpy(), PARENTS)

    for i in range(BATCH_SIZE):
        Path(f'{VIZ_OUTPUT_DIRECTORY}/{i}').mkdir(parents=True, exist_ok=True)

        with open(f'{VIZ_OUTPUT_DIRECTORY}/{i}/ground_truth.json', 'w') as f:
            f.truncate(0)
            f.write(json.dumps(x[i, :, :, :].tolist()))

        with open(f'{VIZ_OUTPUT_DIRECTORY}/{i}/input.json', 'w') as f:
            f.truncate(0)
            f.write(json.dumps(in_x[i, :, :, :].tolist()))

        with open(f'{VIZ_OUTPUT_DIRECTORY}/{i}/output.json', 'w') as f:
            f.truncate(0)
            f.write(json.dumps(out_x[i, :, :, :].tolist()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        help='Name of the model. Used for loading and saving weights.',
        type=str,
        default='default')

    args = parser.parse_args()

    visualize(args.model_name)