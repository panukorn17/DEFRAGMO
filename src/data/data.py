import time
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from utils.file_utils import load_data

class FragmentDataset(Dataset):
    """
    This class creates a dataset from the data, source and target of fragmented molecules.

    Attributes:
    data: the source of the full dataset
    """
    def __init__(self):
        """
        The constructor for the FragmentDataset class.

        Parameters:
        data: the source of the full dataset
        """
        self.data = pd.read_csv('DATA/ZINC/PROCESSED/train.smi').reset_index(drop=True)

if __name__ == '__main__':
    dataset = load_data()
    print(dataset.data.head())