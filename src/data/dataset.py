import time
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from utils.file_utils import load_data
from .fragment_embeddings import Vocabulary

class FragmentDataset(Dataset):
    """
    This class creates a dataset from the data, source and target of fragmented molecules.
    """
    def __init__(self):
        """
        The constructor for the FragmentDataset class.

        Parameters:
        data: the source of the full dataset
        size: the size of the dataset
        vocab: the vocabulary of the dataset
        """
        self.data = load_data()
        self.size = len(self.data)
        self.vocab = None
    
    def __len__(self):
        """
        This method returns the length of the dataset.

        Returns:
        The length of the dataset.
        """
        return self.size
    
    def __getitem__(self, idx):
        """
        This method returns the source and target of a datapoint.

        Parameters:
        idx: the index of the item

        Returns:
        tuple: A tuple containing three elements:
            - src (list): The source sequence of fragments, translated to integers
            - tgt (list): The target sequence of fragments, translated to integers
            - seq (list): The sequence of fragments, in string format, without delimiters
        """
        seq = self.data.fragments[idx].split(" ")
        seq = self.vocab.append_delimiters(seq)
        src = self.vocab.translate(seq[:-1])
        tgt = self.vocab.translate(seq[1:])
        return src, tgt, seq[1:-1]
    
    def set_vocab(self, vocab):
        """
        This method sets the vocabulary of the dataset.

        Parameters:
        vocab: the vocabulary of the dataset
        """
        start = time.time()
        if self.vocab is None:
            self.vocab = Vocabulary.load()
        end = time.time()
        print(f"Time elapsed to set the vocabulary: %H:%M:%S{time.gmtime(end - start)}.")
        return self.vocab