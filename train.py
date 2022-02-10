from constants import *

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
        dim_model=66,
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

    for epoch in range(EPOCHS):
        transformer.train()
        loss_g = 0
        tqdm_dataloader = tqdm(train_dataloader)
        for index, batch in enumerate(tqdm_dataloader):
            seq = batch["X"].reshape((BATCH_SIZE, WINDOW_SIZE, 66))
            out = transformer(seq, seq)
            optimizer_g.zero_grad()
            loss = criterion(seq, out)
            loss.backward()
            optimizer_g.step()
            tqdm_dataloader.set_description(f"batch: {index + 1} loss: {loss}")
            loss_g += loss

        print(f"epoch: {epoch + 1}, loss: {loss_g}")
        

if __name__ == '__main__':
    train()