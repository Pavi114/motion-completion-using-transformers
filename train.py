from constants import LAFAN1_DIRECTORY

from util import load_data
# Define hyperparameters

# Load and Preprocess Data
l = load_data.load_dataset(LAFAN1_DIRECTORY)
print(l[0])
# Training Loop

# Testing