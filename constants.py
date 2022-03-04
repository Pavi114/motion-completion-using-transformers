import torch

from hyperparameters import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIXED_POINTS = list(range(0, WINDOW_SIZE, KEYFRAME_GAP))

if (WINDOW_SIZE - 1) % KEYFRAME_GAP != 0:
    FIXED_POINTS.append(WINDOW_SIZE - 1)

FIXED_POINTS = torch.LongTensor(FIXED_POINTS).to(DEVICE)

DIM_MODEL = NUM_JOINTS * Q_EMBEDDING_DIM + P_EMBEDDING_DIM + V_EMBEDDING_DIM