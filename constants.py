LAFAN1_DIRECTORY='data/lafan1'
FILES_TO_READ = 5
NUM_JOINTS = 22

BATCH_SIZE = 8
WINDOW_SIZE = 32

NUM_WORKERS = 4

NUM_HEADS = 2
NUM_ENCODER = 1
NUM_DECODER = 1
DROPOUT_P = 0.2

Q_EMBEDDING_DIM = 6
P_EMBEDDING_DIM = 6
V_EMBEDDING_DIM = 6

DIM_MODEL = NUM_JOINTS * Q_EMBEDDING_DIM + P_EMBEDDING_DIM + V_EMBEDDING_DIM


LEARNING_RATE = 0.01
EPOCHS = 5