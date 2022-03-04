import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAFAN1_DIRECTORY='data/lafan1'
FILES_TO_READ = 1 # Set to -1 to read all.
NUM_JOINTS = 22
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

BATCH_SIZE = 8
WINDOW_SIZE = 64
KEYFRAME_GAP = 16

NUM_WORKERS = 4

NUM_HEADS = 1
NUM_ENCODER = 1
NUM_DECODER = 1
DROPOUT_P = 0.2

Q_EMBEDDING_DIM = 3
P_EMBEDDING_DIM = 3
V_EMBEDDING_DIM = 3

LEARNING_RATE = 1e-3
EPOCHS = 5

MODEL_SAVE_DIRECTORY='saved_weights'
VIZ_OUTPUT_DIRECTORY='output'