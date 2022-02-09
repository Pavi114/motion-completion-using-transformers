from constants import *

from torch.optim import Adam
from tqdm import tqdm
from util.load_data import load_dataset
from model.transformer import Transformer

# Define hyperparameters

# Load and Preprocess Data
train_dataloader, test_dataloader = load_dataset(LAFAN1_DIRECTORY)

example_batch = next(iter(train_dataloader))
example_batch["X"] = example_batch["X"].reshape((BATCH_SIZE, WINDOW_SIZE, 66))
print(example_batch["X"].shape)

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

for epoch in range(EPOCHS):
    transformer.train()
    for index, batch in enumerate(tqdm(train_dataloader)):
        pass
        # call transformer



# Testing
