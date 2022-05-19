import argparse
from itertools import chain
from random import choices
from constants import DEVICE, MODEL_SAVE_DIRECTORY

import torch
from torch.nn import L1Loss
from torch.optim import Adam
from tqdm import tqdm
from model.encoding.input_encoder import InputEncoder
from model.encoding.output_decoder import OutputDecoder
from model.loss.fk_loss import FKLoss
from train_stats import load_stats
from util.interpolation.fixed_points import get_fixed_points
from util.interpolation.interpolation_factory import get_p_interpolation, get_q_interpolation
from util.interpolation.linear_interpolation import single_linear_interpolation
from util.interpolation.spherical_interpolation import single_spherical_interpolation
from util.load_data import load_train_dataset
from model.transformer import Transformer
from util.math import round_tensor
from util.read_config import read_config
from util.plot import plot_loss


def train(model_name='default', save_weights=False, load_weights=False):
    # Load config
    config = read_config(model_name)

    # Load and Preprocess Data
    train_dataloader = load_train_dataset(config['dataset'])

    # Training Loop
    transformer = Transformer(config).to(DEVICE)

    input_encoder = InputEncoder(config['embedding_size']).to(DEVICE)

    output_decoder = OutputDecoder(config['embedding_size']).to(DEVICE)

    optimizer_g = Adam(lr=config['hyperparameters']['learning_rate'],
                       params=chain(transformer.parameters(),
                                    input_encoder.parameters(),
                                    output_decoder.parameters()))

    criterion = L1Loss().to(DEVICE)

    fk_criterion = FKLoss().to(DEVICE)

    best_loss = torch.Tensor([float("+inf")]).to(DEVICE)

    loss_history = []

    if load_weights:
        checkpoint = torch.load(
            f'{MODEL_SAVE_DIRECTORY}/model_{model_name}.pt')

        transformer.load_state_dict(checkpoint['transformer_state_dict'])
        input_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        output_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['loss']

    max_keyframe_gap = config['dataset']['max_window_size'] - \
        (config['dataset']['front_pad'] + config['dataset']['back_pad'])

    _, _, offsets, _ = load_stats()
    offsets = torch.Tensor(offsets).to(DEVICE)
    offsets = offsets.repeat((config['dataset']['batch_size'], config['dataset']['max_window_size'], 1, 1))

    for epoch in range(config['hyperparameters']['epochs']):
        transformer.train()
        train_loss = 0
        tqdm_dataloader = tqdm(train_dataloader)

        random_weighted_keyframe_gap = iter(choices(
            population=[i for i in range(5, max_keyframe_gap + 1)],
            weights=[1/i for i in range(5, max_keyframe_gap + 1)],
            k=10000
        ))

        for index, batch in enumerate(tqdm_dataloader):
            local_q = round_tensor(batch["local_q"].to(DEVICE), decimals=4)
            local_p = round_tensor(batch["local_p"].to(DEVICE), decimals=4)
            root_p = round_tensor(
                batch["X"][:, :, 0, :].to(DEVICE), decimals=4)
            root_v = round_tensor(batch["root_v"].to(DEVICE), decimals=4)

            # in_local_q = q_interpolation_function(local_q, 1, fixed_points)
            # in_root_p = p_interpolation_function(root_p, 1, fixed_points)
            # in_root_v = p_interpolation_function(root_v, 1, fixed_points)

            keyframe_gap = next(random_weighted_keyframe_gap)

            in_local_q = single_spherical_interpolation(
                local_q, dim=1, front=config['dataset']['front_pad'], keyframe_gap=keyframe_gap, back=config['dataset']['back_pad'])
            in_root_p = single_linear_interpolation(
                root_p, dim=1, front=config['dataset']['front_pad'], keyframe_gap=keyframe_gap, back=config['dataset']['back_pad'])
            in_root_v = single_linear_interpolation(
                root_v, dim=1, front=config['dataset']['front_pad'], keyframe_gap=keyframe_gap, back=config['dataset']['back_pad'])

            seq = input_encoder(in_local_q, in_root_p, in_root_v)

            out = transformer(seq, seq, keyframe_gap)

            out_q, out_p, out_v = output_decoder(out)

            out_local_p = torch.cat([
                out_p.unsqueeze(dim=2),
                offsets
            ], dim=2)

            optimizer_g.zero_grad()

            q_loss = criterion(local_q, out_q)
            # p_loss = criterion(root_p, out_p)
            # v_loss = criterion(root_v, out_v)
            fk_loss = fk_criterion(local_p, local_q, out_local_p, out_q)

            loss = 10 * q_loss + fk_loss

            loss.backward()

            optimizer_g.step()
            tqdm_dataloader.set_description(
                f"{model_name} | batch: {index + 1} | loss: {loss:.4f} q_loss: {q_loss:.4f} fk_loss: {fk_loss:.4f}"
            )
            train_loss += loss

        loss_history.append(train_loss.detach().cpu().numpy())
        print(f"epoch: {epoch + 1}, train loss: {train_loss/index}")

        if save_weights and train_loss < best_loss:
            # Save weights
            torch.save(
                {
                    'transformer_state_dict': transformer.state_dict(),
                    'encoder_state_dict': input_encoder.state_dict(),
                    'decoder_state_dict': output_decoder.state_dict(),
                    'optimizer_state_dict': optimizer_g.state_dict(),
                    'loss': best_loss
                }, f'{MODEL_SAVE_DIRECTORY}/model_{model_name}.pt')

            best_loss = train_loss

    plot_loss(loss_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        help='Name of the model. Used for loading and saving weights.',
        type=str,
        default='default')

    parser.add_argument('--save_weights',
                        help='Save model weights.',
                        action=argparse.BooleanOptionalAction,
                        default=False)

    parser.add_argument('--load_weights',
                        help='Load model weights.',
                        action=argparse.BooleanOptionalAction,
                        default=False)

    args = parser.parse_args()

    train(args.model_name, args.save_weights, args.load_weights)
