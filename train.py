from itertools import chain
import json
from random import choice
from constants import *

import torch
from torch.nn import L1Loss
from torch.optim import Adam
from tqdm import tqdm
from model.encoding.input_encoder import InputEncoder
from model.encoding.output_decoder import OutputDecoder
from model.loss.fk_loss import FKLoss
from util.load_data import load_dataset
from util.quaternions import quat_fk
from model.transformer import Transformer

def train():
    # Define hyperparameters

    # Load and Preprocess Data
    train_dataloader, test_dataloader = load_dataset(LAFAN1_DIRECTORY)

    # Training Loop
    transformer = Transformer(
        dim_model=DIM_MODEL,
        num_heads=NUM_HEADS,
        seq_len=WINDOW_SIZE,
        num_encoder_layers=NUM_ENCODER,
        num_decoder_layers=NUM_DECODER,
        dropout_p=DROPOUT_P,
    )

    input_encoder = InputEncoder()

    output_decoder = OutputDecoder()

    optimizer_g = Adam(
        lr=LEARNING_RATE,
        params=chain(
            transformer.parameters(),
            input_encoder.parameters(),
            output_decoder.parameters()
        )
    )

    criterion = L1Loss()

    fk_criterion = FKLoss()

    for epoch in range(EPOCHS):
        transformer.train()
        train_loss = 0
        tqdm_dataloader = tqdm(train_dataloader)
        for index, batch in enumerate(tqdm_dataloader):
            local_q = batch["local_q"]
            local_p = batch["local_p"]
            root_p = batch["X"][:,:,0,:]
            root_v = batch["root_v"]

            seq = input_encoder(local_q, root_p, root_v)

            out = transformer(seq, seq)

            out_q, out_p, out_v = output_decoder(out)

            out_local_p = local_p
            out_local_p[:, :, 0, :] = out_p

            optimizer_g.zero_grad()

            q_loss = criterion(local_q, out_q)
            # p_loss = criterion(root_p, out_p)
            # v_loss = criterion(root_v, out_v)
            fk_loss = fk_criterion(local_p, local_q, out_local_p, out_q)

            loss = q_loss + fk_loss

            loss.backward()

            optimizer_g.step()
            tqdm_dataloader.set_description(f"batch: {index + 1} loss: {loss} q_loss: {q_loss} fk_loss: {fk_loss}")
            train_loss += loss

        print(f"epoch: {epoch + 1}, train loss: {train_loss/index}")

        # Visualize
        viz_batch = next(iter(train_dataloader))

        local_q = viz_batch["local_q"][:1, :, :, :]
        local_p = viz_batch["local_p"][:1, :, :, :]
        root_p = viz_batch["X"][:1, :, 0,:]
        root_v = viz_batch["root_v"][:1, :, :]

        seq = input_encoder(local_q, root_p, root_v)

        out = transformer(seq, seq)

        out_q, out_p, out_v = output_decoder(out)

        out_local_p = local_p
        out_local_p[:, :, 0, :] = out_p

        _, x = quat_fk(local_q.detach().numpy(), local_p.detach().numpy(), PARENTS)
        _, out_x = quat_fk(out_q.detach().numpy(), out_local_p.detach().numpy(), PARENTS)

        with open('./viz/dist/static/animations/ground_truth.json', 'w') as f:
            f.truncate(0)
            f.write(json.dumps(x[0, :, :, :].tolist()))
        
        with open('./viz/dist/static/animations/output.json', 'w') as f:
            f.truncate(0)
            f.write(json.dumps(out_x[0, :, :, :].tolist()))      

if __name__ == '__main__':
    train()