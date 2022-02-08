from typing import Tuple
from torch.utils.data import DataLoader

from util import lafan1

from constants import BATCH_SIZE, NUM_WORKERS

def load_dataset(dataset_directory: str) -> Tuple[DataLoader, DataLoader]:
    """Function to load dataset from the given directory, perform pre-processing.

    Args:
        dataset_directory (str): Location of the dataset

    Returns:
        Tuple[DataLoader, DataLoader]: (train_dataloader, test_dataloader)
    """
    lafan_train_dataset = lafan1.LaFan1(dataset_directory=dataset_directory, train=True)
    lafan_train_loader = DataLoader(lafan_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    

    lafan_test_dataset = lafan1.LaFan1(dataset_directory=dataset_directory, train=False)
    lafan_test_loader = DataLoader(lafan_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    return (lafan_train_loader, lafan_test_loader)

