from constants import *

import torch
from torch.nn import Sequential, Linear, Dropout, ReLU
from torch.optim import Adam
from tqdm import tqdm
from model.loss.l1_loss import L1Loss
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

    optimizer_g = Adam(
        lr=LEARNING_RATE,
        params=list(transformer.parameters())
    )

    criterion = L1Loss()

    q_encoder = Sequential(Linear(4, 16), Dropout(0.1), ReLU(), Linear(16, 8), Dropout(0.1))

    q_decoder = Sequential(Linear(8, 16), Dropout(0.1), ReLU(), Linear(16, 4), Dropout(0.1))

    for epoch in range(EPOCHS):
        transformer.train()
        loss_g = 0
        tqdm_dataloader = tqdm(train_dataloader)
        for index, batch in enumerate(tqdm_dataloader):
            local_q = batch["local_q"]
            # root_p = batch["X"][:,:,0,:]
            # root_v = batch["root_v"]

            embedded_q = q_encoder(local_q).reshape((BATCH_SIZE, WINDOW_SIZE, 176))

            # seq = torch.cat([local_q, root_p, root_v], axis=-1)
            seq = embedded_q

            out = transformer(seq, seq)

            out = out.reshape((BATCH_SIZE, WINDOW_SIZE, 22, 8))

            decoded_out = q_decoder(out)

            optimizer_g.zero_grad()
            loss = criterion(local_q, decoded_out)
            loss.backward()
            optimizer_g.step()
            tqdm_dataloader.set_description(f"batch: {index + 1} loss: {loss}")
            loss_g += loss

        print(f"epoch: {epoch + 1}, loss: {loss_g}")
    
    print(decoded_out)
        

if __name__ == '__main__':
    train()