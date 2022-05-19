from typing import Tuple
from torch.utils.data import DataLoader
from constants import LAFAN1_DIRECTORY

from util.lafan1 import LaFan1


def load_train_dataset(dataset_config) -> DataLoader:
    """Function to load dataset from the given directory, perform pre-processing.

    Args:
        dataset_directory (str): Location of the dataset

    Returns:
        DataLoader: train_dataloader
    """
    lafan_train_dataset = LaFan1(dataset_directory=LAFAN1_DIRECTORY,
                                 train=True,
                                 seq_len=dataset_config['max_window_size'],
                                 files_to_read=dataset_config['files_to_read'])
    lafan_train_loader = DataLoader(
        lafan_train_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=True,
        num_workers=dataset_config['num_workers'],
        drop_last=True,
    )

    return lafan_train_loader


def load_viz_dataset(dataset_config) -> DataLoader:
    """Function to load dataset from the given directory, perform pre-processing.

    Args:
        dataset_directory (str): Location of the dataset

    Returns:
        DataLoader: viz_dataloader
    """
    lafan_viz_dataset = LaFan1(dataset_directory=LAFAN1_DIRECTORY,
                               train=False,
                               seq_len=dataset_config['max_window_size'],
                               files_to_read=1)
    lafan_viz_loader = DataLoader(
        lafan_viz_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=False,
        num_workers=dataset_config['num_workers'],
        drop_last=True,
    )

    return lafan_viz_loader


def load_test_dataset(dataset_config) -> DataLoader:
    """Function to load dataset from the given directory, perform pre-processing.

    Args:
        dataset_directory (str): Location of the dataset

    Returns:
        DataLoader: test_dataloader
    """
    lafan_test_dataset = LaFan1(dataset_directory=LAFAN1_DIRECTORY,
                                train=False,
                                seq_len=65,
                                offset=40,
                                files_to_read=-1)
                                
    lafan_test_loader = DataLoader(
        lafan_test_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=False,
        num_workers=dataset_config['num_workers'],
        drop_last=True,
    )

    return lafan_test_loader
