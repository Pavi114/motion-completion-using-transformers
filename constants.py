import torch

LAFAN1_DIRECTORY='data/lafan1'
NUM_JOINTS = 22
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_DIRECTORY='saved_weights'
VIZ_OUTPUT_DIRECTORY='output'