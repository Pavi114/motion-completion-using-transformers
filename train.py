import argparse
from itertools import chain
from constants import DEVICE, MODEL_SAVE_DIRECTORY

import torch
from torch.nn import L1Loss
from torch.optim import Adam
from tqdm import tqdm
from model.encoding.input_encoder import InputEncoder
from model.encoding.output_decoder import OutputDecoder
from model.loss.fk_loss import FKLoss
from util.interpolation.fixed_points import get_fixed_points
from util.interpolation.interpolation_factory import get_p_interpolation, get_q_interpolation
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
    
    p_interpolation_function = get_p_interpolation(config['hyperparameters']['interpolation'])
    q_interpolation_function = get_q_interpolation(config['hyperparameters']['interpolation'])

    fixed_points = get_fixed_points(config['dataset']['window_size'], config['dataset']['keyframe_gap'])

    if load_weights:
        checkpoint = torch.load(
            f'{MODEL_SAVE_DIRECTORY}/model_{model_name}.pt')

        transformer.load_state_dict(checkpoint['transformer_state_dict'])
        input_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        output_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['loss']

    for epoch in range(config['hyperparameters']['epochs']):
        transformer.train()
        train_loss = 0
        tqdm_dataloader = tqdm(train_dataloader)
        for index, batch in enumerate(tqdm_dataloader):
            local_q = round_tensor(batch["local_q"].to(DEVICE), decimals=4)
            local_p = round_tensor(batch["local_p"].to(DEVICE), decimals=4)
            root_p = round_tensor(batch["X"][:, :, 0, :].to(DEVICE), decimals=4)
            root_v = round_tensor(batch["root_v"].to(DEVICE), decimals=4)

            in_local_q = q_interpolation_function(local_q, 1, fixed_points)
            in_root_p = p_interpolation_function(root_p, 1, fixed_points)
            in_root_v = p_interpolation_function(root_v, 1, fixed_points)

            # print(torch.any(torch.isnan(in_local_q)))

            # print(local_q[0][0])
            # print(local_q[0][3])
            # print(in_local_q[0][1])

            # seq = input_encoder(local_q, root_p, root_v)
            seq = input_encoder(in_local_q, in_root_p, in_root_v)

            out = transformer(seq, seq)

            # print(out)

            # break

            out_q, out_p, out_v = output_decoder(out)

            out_local_p = local_p
            out_local_p[:, :, 0, :] = out_p

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
