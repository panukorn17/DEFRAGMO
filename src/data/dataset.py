import time
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from utils.file_utils import load_data
from data.fragment_embeddings import Vocabulary

class DataCollator:
    """
    This class creates a data collator for the DataLoader.
    """
    def __init__(self, vocab):
        """
        The constructor for the DataCollator class.

        Parameters:
        vocab: the vocabulary of the dataset
        """
        self.vocab = vocab
    
    def stack_sequences(self, fragment_sequences):
        """
        This method stacks sequences of fragments into a 2D array with PAD tokens to ensure all sequences 
        have the same length (the longest sequence). The 2D array is then converted to a 2D tensor. 

        Parameters:
        fragment_sequences (list of list of int) [batch_size, [sequence_length]]: the sequences of fragment identifiers

        Returns:
        tuple: A tuple containing two elements:
            - padded_seqs (torch.Tensor) [batch_size, longest_sequence_lenth]: the stacked sequences of fragments identified by their indices
            - lengths (list of int) [batch_size]: the lengths of the sequences
        """
        # Sort the sequences in the batch by length in descending order
        sorted_fragment_sequences = sorted(fragment_sequences, key=len, reverse=True)
        # Get the maximum length of sequences
        max_length = len(sorted_fragment_sequences[0])
        # Initialise a 2D array of PAD tokens of shape [batch_size, longest_sequence_lenth] to fill with fragment sequences in the batch
        padded_seqs = np.full((len(sorted_fragment_sequences), max_length), self.vocab.PAD)
        # Fill the padded array with fragment sequences in the batch
        for i, seq in enumerate(sorted_fragment_sequences):
            padded_seqs[i, :len(seq)] = seq[:len(seq)]
        # Convert the padded array to a 2D tensor
        padded_seqs = torch.tensor(padded_seqs, dtype=torch.long)
        # Get the lengths of sequences
        lengths = [len(seq) for seq in sorted_fragment_sequences]
        return padded_seqs, lengths
    
    def __call__(self, batch):
        """
        This method collates the batch of data.

        Parameters:
        batch (list of tuple): The batch of data, where each tuple contains three elements:
            - src (list of list of int) [batch_size, [sequence_length]]: The source sequences of fragments,
                represented as a list of integers. The 'sequence_length' can vary for each sequence, and 'batch_size'
                denotes the number of sequences. Includes the SOS (start of sequence) token. 
                Example: [1, 424, 136, 560, 630].
            - tgt (list of list of int) [batch_size, [sequence_length]]: The target sequences of fragments,
                similar to the source sequences but including the EOS (end of sequence) token. 
                Example: [424, 136, 560, 630, 2].
            - seq (list of list of str) [batch_size, [sequence_length]]: The sequences of fragments in string format,
                similar to the source sequences but without EOS or SOS tokens, and representing each fragment in string format. 
                Example: ['*CCCC', '*C1CC(=O)N(*)C1', '*C(*)=O', '*N1CC[NH+](C)CC1'].

        Returns:
        tuple: A tuple containing two elements:
            - src_tensor (tensor) [batch_size, longest_sequence_length]: The source sequence of fragment identifiers, with PAD tokens.
                Example: tensor([[  1, 266, 571, 427, 173, 429,  37, 490, 354],
                                 [  1, 424, 490,  37, 560, 571, 613, 581,   0],
                                 [  1, 424, 136, 560, 566, 429, 566,   0,   0],
                                 [  1, 208, 403,  37, 490, 515,   0,   0,   0],
                                 [  1, 637, 490, 427, 206,   0,   0,   0,   0],
                                 [  1, 409, 618, 414,   0,   0,   0,   0,   0]])
            - tgt_tensor (tensor) [batch_size, longest_sequence_length]: The target sequence of fragment identifiers, with PAD tokens.
                Example: tensor([[266, 571, 427, 173, 429,  37, 490, 354,   2],
                                 [424, 490,  37, 560, 571, 613, 581,   2,   0],
                                 [424, 136, 560, 566, 429, 566,   2,   0,   0],
                                 [208, 403,  37, 490, 515,   2,   0,   0,   0],
                                 [637, 490, 427, 206,   2,   0,   0,   0,   0],
                                 [409, 618, 414,   2,   0,   0,   0,   0,   0]])
            - src_lengths (list of int) [batch_size]: The lengths of the source sequences of fragments.
                Example: [9, 8, 7, 6, 5, 4]
            - idx (list of int) [batch_size]: The indices of the dataset.
        """
        
        # Unzip the batch into source and target sequences
        src, tgt, idx, seq = zip(*batch)
        # Merge the sequences of fragments
        src_tensor, src_lengths = self.stack_sequences(src)
        tgt_tensor, tgt_lengths = self.stack_sequences(tgt)
        return src_tensor, tgt_tensor, src_lengths, idx

class MoleculeFragmentsDataset(Dataset):
    """
    This class represents the dataset of molecule fragments. The MoleculeFragmentsDataset.Dataset call is compatible with the torch.utils.data.DataLoader class.
    It is responsible for loading and batching of the fragments dataset for training the model. Specifically, it handles sequence-to-sequence data by providing
    the source and target sequences of fragments when training the model.
    """
    def __init__(self, config):
        """
        The constructor for the FragmentDataset class. This sets the data, vocabulary and data parameters according to 
        the run configuration.

        Parameters:
        config: the source of the full dataset
        size: the size of the dataset
        vocab: the vocabulary of the dataset
        """
        self.config = config
        self.data = load_data(self.config, data_type = "train")
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
            - idx (list): The index of the dataset
            - seq (list): The sequence of fragments, in string format, without delimiters
        """
        seq = self.data.fragments[idx].split(" ")
        seq = self.vocab.append_delimiters(seq)
        src = self.vocab.translate(seq[:-1])
        tgt = self.vocab.translate(seq[1:])
        return src, tgt, idx, seq[1:-1]
    
    def set_vocab(self):
        """
        This method sets the vocabulary of the dataset.

        Parameters:
        vocab: the vocabulary of the dataset
        """
        start = time.time()
        if self.vocab is None:
            try:
                self.vocab = Vocabulary.load(self.config)
            except FileNotFoundError:
                print("Vocabulary not found. Creating a new one.")
                self.vocab = Vocabulary(self.config, self.data)
        end = time.time()
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(end - start))
        print(f"Time elapsed to set the vocabulary: {formatted_time}.")
        return self.vocab
    
    def get_loader(self):
        """
        This method returns a DataLoader object for the dataset.

        Returns:
        DataLoader: A DataLoader object for the dataset
        """
        start = time.time()
        collator = DataCollator(self.vocab)
        loader = DataLoader(dataset=self, 
                            batch_size=16, 
                            shuffle=True, 
                            collate_fn=collator)
        end = time.time()
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(end - start))
        print(f"Time elapsed to get the DataLoader: {formatted_time}.")
        return loader