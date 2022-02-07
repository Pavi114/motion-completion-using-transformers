from typing import Tuple
from torch.utils.data import DataLoader

def load_dataset(dataset_directory: str) -> Tuple[DataLoader, DataLoader]:
    """Function to load dataset from the given directory, perform pre-processing.

    Args:
        dataset_directory (str): Location of the dataset

    Returns:
        Tuple[DataLoader, DataLoader]: (train_dataloader, test_dataloader)
    """
    pass