from itertools import chain
from constants import *

import torch
from torch.nn import L1Loss
from torch.optim import Adam
from tqdm import tqdm
from model.encoding.input_encoder import InputEncoder
from model.encoding.output_decoder import OutputDecoder
from util.load_data import load_dataset
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

    for epoch in range(EPOCHS):
        transformer.train()
        loss_g = 0
        tqdm_dataloader = tqdm(train_dataloader)
        for index, batch in enumerate(tqdm_dataloader):
            local_q = batch["local_q"]
            root_p = batch["X"][:,:,0,:]
            root_v = batch["root_v"]

            seq = input_encoder(local_q, root_p, root_v)

            out = transformer(seq, seq)

            out_q, out_p, out_v = output_decoder(out)

            optimizer_g.zero_grad()

            q_loss = criterion(local_q, out_q)
            p_loss = criterion(root_p, out_p)
            v_loss = criterion(root_v, out_v)

            loss = q_loss + p_loss + v_loss

            loss.backward()

            optimizer_g.step()
            tqdm_dataloader.set_description(f"batch: {index + 1} loss: {loss} q_loss: {q_loss} p_loss: {p_loss} v_loss: {v_loss}")
            loss_g += loss

        print(f"epoch: {epoch + 1}, loss: {loss_g}")
    
    print(out_q)
    print(out_p)
    print(out_v)
        

if __name__ == '__main__':
    train()