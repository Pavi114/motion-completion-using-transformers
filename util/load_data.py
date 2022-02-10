from typing import Tuple
from constants import WINDOW_SIZE
from torch.utils.data import DataLoader

from . import lafan1

from constants import BATCH_SIZE, NUM_WORKERS


def load_dataset(dataset_directory: str) -> Tuple[DataLoader, DataLoader]:
    """Function to load dataset from the given directory, perform pre-processing.

    Args:
        dataset_directory (str): Location of the dataset

    Returns:
        Tuple[DataLoader, DataLoader]: (train_dataloader, test_dataloader)
    """
    lafan_train_dataset = lafan1.LaFan1(
        dataset_directory=dataset_directory, train=True, seq_len=WINDOW_SIZE
    )
    lafan_train_loader = DataLoader(
        lafan_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    # lafan_test_dataset = lafan1.LaFan1(
    #     dataset_directory=dataset_directory, train=False, seq_len=WINDOW_SIZE, drop_last=True
    # )
    # lafan_test_loader = DataLoader(
    #     lafan_test_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=NUM_WORKERS,
    #     drop_last=True,
    # )

    return (lafan_train_loader, None)
