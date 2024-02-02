import time
import numpy as np
import pandas as pd

from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from collections import defaultdict
from tqdm import tqdm
from utils.file_utils import save_pickle, load_pickle
from utils.mol_utils import mols_from_smiles, mols_to_smiles

SOS_TOKEN = "<SOS>"
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
TOKENS = [SOS_TOKEN, PAD_TOKEN, EOS_TOKEN]
START_IDX = len(TOKENS)

class Vocabulary:
    """
    This class creates a vocabulary of fragments from the data.
    """
    def __init__(self, data):
        """
        The constructor for the Vocabulary class.

        Parameters:
        data: the source of the full dataset
        """
        w2i, i2w = get_embeddings(data)
        self.w2i = w2i
        self.i2w = i2w
        self.size = len(w2i)
        self.pretrained_model = load_model()
    
    def get_size(self):
        """
        This method returns the size of the vocabulary.

        Returns:
        The size of the vocabulary.
        """
        return self.size
    
    def get_effective_size(self):
        """
        This method returns the effective size of the vocabulary.

        Returns:
        The effective size of the vocabulary (not considering the tokens).
        """
        return self.w2i

    def get_embeddings(self, data):
        """
        This method returns the embeddings of the vocabulary.

        Parameters:
        data: the source of the full dataset

        Returns:
        tuple: A tuple containing two elements:
            - w2i (dict): A dictionary that maps words to integers
            - i2w (dict): A dictionary that maps integers to words
        """
        # Initialise the dictionaries with the special tokens
        w2i = {token: i for i, token in enumerate(TOKENS)}
        i2w = {i: token for i, token in enumerate(TOKENS)}

        # Get a list of unique fragments
        fragments = list(set([frag for molecule_fragments in tqdm(data.fragments, desc="Getting unique fragments from dataset...") for frag in molecule_fragments]))
        
        # Update the dictionaries with the unique fragments
        w2i.update({frag: i + START_IDX for i, frag in enumerate(fragments)})
        i2w.update({i + START_IDX: frag for i, frag in enumerate(fragments)})

        # Convert unique fragments into mol objects
        fragments_mol = mols_from_smiles(fragments)

        # Get the embeddings of the fragments
        print("Getting embeddings for unique fragments...")
        start = time.time()
        embeddings_mol = sentences2vec([mol2alt_sentence(mol, 1) for mol in fragments_mol], self.pretrained_model, unseen='UNK')
        embeddings_mol_vec = np.array([embed.vec for embed in embeddings_mol])
        embeddings_tokens = np.random.uniform(-0.05, 0.05, (len(TOKENS), 100))
        embeddings = np.vstack([embeddings_tokens, embeddings_mol_vec])
        np.savetxt('emb_100.dat', embeddings, delimiter=",")
        end = time.time()
        print(f"Time elapsed to get the embeddings: %H:%M:%S{time.gmtime(end - start)}.")
        return w2i, i2w
    
    def load_model(self):
        """
        This method loads the pretrained model.

        Returns:
        model: The pretrained model
        """
        print("Loading the pretrained model...")
        start = time.time()
        model = load_pickle("data/mol2vec_model.pkl")
        end = time.time()
        print(f"Time elapsed to load the pretrained model: %H:%M:%S{time.gmtime(end - start)}.")
        return model