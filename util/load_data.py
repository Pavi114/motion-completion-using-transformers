from typing import Tuple
from hyperparameters import FILES_TO_READ, WINDOW_SIZE
from torch.utils.data import DataLoader

from util.lafan1 import LaFan1

from hyperparameters import BATCH_SIZE, NUM_WORKERS


def load_train_dataset(dataset_directory: str) -> DataLoader:
    """Function to load dataset from the given directory, perform pre-processing.

    Args:
        dataset_directory (str): Location of the dataset

    Returns:
        DataLoader: train_dataloader
    """
    lafan_train_dataset = LaFan1(
        dataset_directory=dataset_directory, train=True, seq_len=WINDOW_SIZE, files_to_read=FILES_TO_READ
    )
    lafan_train_loader = DataLoader(
        lafan_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    return lafan_train_loader

def load_viz_dataset(dataset_directory: str) -> DataLoader:
    """Function to load dataset from the given directory, perform pre-processing.

    Args:
        dataset_directory (str): Location of the dataset

    Returns:
        DataLoader: viz_dataloader
    """
    lafan_viz_dataset = LaFan1(
        dataset_directory=dataset_directory, train=False, seq_len=WINDOW_SIZE, files_to_read=1
    )
    lafan_viz_loader = DataLoader(
        lafan_viz_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    return lafan_viz_loader

def load_test_dataset(dataset_directory: str) -> DataLoader:
    """Function to load dataset from the given directory, perform pre-processing.

    Args:
        dataset_directory (str): Location of the dataset

    Returns:
        DataLoader: test_dataloader
    """
    lafan_test_dataset = LaFan1(
        dataset_directory=dataset_directory, train=False, seq_len=WINDOW_SIZE, files_to_read=FILES_TO_READ
    )
    lafan_test_loader = DataLoader(
        lafan_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    return lafan_test_loader
