import time
import numpy as np
import pandas as pd

from gensim.models.word2vec import Word2Vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from collections import defaultdict
from tqdm import tqdm
from utils.file_utils import save_pickle, load_pickle
from utils.mol_utils import mols_from_smiles, mols_to_smiles
from utils.config import Config

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
TOKENS = [PAD_TOKEN, SOS_TOKEN , EOS_TOKEN]
START_IDX = len(TOKENS)

class Vocabulary:
    """
    This class creates a vocabulary of fragments from the data.
    """
    @classmethod
    def load(cls, config: Config) -> 'Vocabulary':
        """
        This method loads the vocabulary from the config directory.

        Parameters:
        config (Config): the configuration of the run.

        Returns:
        Vocabulary: The vocabulary of fragments.
        """
        print("Loading the vocabulary...")
        path = config.path('config')
        vocab = load_pickle(path / 'vocab.pkl')
        print(f"Vocabulary loaded from {path}.")
        return vocab

    def save(self, config: Config) -> None:
        """
        This method saves the vocabulary to the config directory.

        Parameters:
        config (Config): the configuration of the run.
        """
        print("Saving the vocabulary...")
        path = config.path('config')
        save_pickle(self, path / 'vocab.pkl')
        print(f"Vocabulary saved to {path}.")

    def __init__(self, config: Config, data:pd.DataFrame):
        """
        The constructor for the Vocabulary class.

        Parameters:
        config (Config): The configuration of the run.
        data (pd.DataFrame) [feature_length, number_of_molecules]: the source of the full dataset
        """
        self.config = config
        self.pretrained_model = self.load_model()
        w2i, i2w = self.get_embeddings(self.config, data)
        self.w2i = w2i
        self.i2w = i2w
        self.size = len(w2i)

        self.save(config)
    
    def get_size(self) -> int:
        """
        This method returns the size of the vocabulary.

        Returns:
        The size of the vocabulary [number_of_unique_fragments].
        """
        return self.size
    
    def get_effective_size(self) -> int:
        """
        This method returns the effective size of the vocabulary.

        Returns:
        The effective size of the vocabulary not considering the special tokens [number_of_unique_fragments - 3].
        """
        return self.w2i
    
    def append_delimiters(self, sentence) -> list:
        """
        This method appends SOS and EOS tokens onto the sentence fragments.
        
        Parameters:
        sentence (list of strings) [sequence_length]: The sequence of fragments that make up the molecule.
        Example: ['*CC', '*Cc1ccccn1', '*N(*)*', '*C(*)=O', '*N*', '*C*', '*c1ccc(*)cc1', '*C(C)(C)C']

        Returns:
        sentence_with_delimiters (list of strings) [sequence_length + 2]:The sequence of fragments that make up the molecule
        including the SOS and EOS tokents.
        Example: ['<SOS>', '*CC', '*Cc1ccccn1', '*N(*)*', '*C(*)=O', '*N*', '*C*', '*c1ccc(*)cc1', '*C(C)(C)C', '<EOS>']
        """
        sentence_with_delimiters = [SOS_TOKEN] + sentence + [EOS_TOKEN]
        return sentence_with_delimiters
    
    @property
    def SOS(self) -> int:
        """
        This method returns the index of the SOS token as an integer. This is a method that behaves like an attribute.
        When calling this method, simply use the syntax: vocab.SOS.

        Returns:
        The index of the SOS token as an integer.
        """
        return self.w2i[SOS_TOKEN]
    
    @property
    def PAD(self) -> int:
        """
        This method returns the index of the PAD token as an integer. This is a method that behaves like an attribute.
        When calling this method, simply use the syntax: vocab.PAD.

        Returns:
        The index of the PAD token as an integer.
        """
        return self.w2i[PAD_TOKEN]
    
    @property
    def EOS(self) -> int:
        """
        This method returns the index of the EOS token as an integer. This is a method that behaves like an attribute.
        When calling this method, simply use the syntax: vocab.EOS.

        Returns:
        The index of the EOS token as an integer.
        """
        return self.w2i[EOS_TOKEN]
    
    @property
    def TOKEN_IDS(self)->list:
        """
        This method returns the indices of the special tokens as a list. This is a method that behaves like an attribute.
        When calling this method, simply use the syntax: vocab.TOKEN_IDS.
        """
        return [self.SOS, self.EOS, self.PAD]
    
    def get_embeddings(self, config, data) -> tuple:
        """
        This method returns the embeddings of the vocabulary.

        Parameters:
        config (Config): the configuration of the run
        data (pd.DataFrame): the dataset to create the vocabulary from

        Returns:
        tuple: A tuple containing two elements:
            - w2i (dict) [number_of_unique_fragments]: A dictionary that maps words to integers
            - i2w (dict) [number_of_unique_fragments]: A dictionary that maps integers to words
        """
        # Initialise the dictionaries with the special tokens
        w2i = {token: i for i, token in enumerate(TOKENS)}
        i2w = {i: token for i, token in enumerate(TOKENS)}

        # Get a list of unique fragments
        fragments = list(set([frag for molecule_fragments in tqdm(data.fragments, desc="Getting unique fragments from dataset...") for frag in molecule_fragments.split()]))
        
        # Update the dictionaries with the unique fragments
        w2i.update({frag: i + START_IDX for i, frag in enumerate(fragments)})
        i2w.update({i + START_IDX: frag for i, frag in enumerate(fragments)})

        # Convert unique fragments into mol objects
        fragments_mol = mols_from_smiles(fragments)

        # Get the embeddings of the fragments
        print("Getting embeddings for unique fragments...")

        start = time.time()
        
        # Constructing sentences
        fragment_sentence = [MolSentence(mol2alt_sentence(frag_mol, 1)) for frag_mol in fragments_mol]
        
        # Extracting embeddings to a numpy.array
        # Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures
        embeddings_mol = [DfVec(x) for x in sentences2vec(fragment_sentence, self.pretrained_model, unseen='UNK')]
        embeddings_mol_vec = np.array([embed.vec for embed in embeddings_mol])
        embeddings_tokens = np.random.uniform(-0.05, 0.05, (len(TOKENS), 100))
        embeddings = np.vstack([embeddings_tokens, embeddings_mol_vec])
        
        # Save the embeddings
        config_dir = config.path('config')
        np.savetxt(f'{config_dir}/emb_100.dat', embeddings, delimiter=",")
        end = time.time()
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(end - start))
        print(f"Time elapsed to get the embeddings: {formatted_time}.")
        return w2i, i2w

    def load_model(self) -> Word2Vec:
        """
        This method loads the pretrained model.

        Returns:
        model (Word2Vec): The pretrained model
        """
        print("Loading the pretrained model...")
        start = time.time()
        model = Word2Vec.load(f"{self.config.path('pretrained').as_posix()}/model_300dim.pkl")
        end = time.time()
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(end - start))
        print(f"Time elapsed to load the pretrained model: {formatted_time}.")
        return model
    
    def translate_str(self, word) -> int:
        """
        This method translates a fragment from a string to an integer.

        Parameters:
        word (string) [1]: The fragment that makes a the molecule. This includes SOS and EOS tokens.
        Example: word = '*c1ccc(C)cc1'

        Returns:
        word_translated (integer) [1]: The fragment that makes up the molecule translated to an integer.
        This would inclued the SOS and EOS tokens.
        Example: translate_str(word) = 575
        """
        word_translated = self.w2i[word]
        return word_translated

    def translate_int(self, word) -> str:
        """
        This method translates a fragment from an integer to a string.

        Parameters:
        word (integer) [1]: The fragment that makes up the molecule translated to an integer.
        This would inclued the SOS and EOS tokens.
        Example: word = 575

        Returns:
        word_translated (string) [1]: The fragment that makes a the molecule. This includes SOS and EOS tokens.
        Example: translate_int(word) = '*c1ccc(C)cc1'
        """
        word_translated = self.i2w[word]
        return word_translated
    
    def translate(self, sentence) -> list:
        """
        This method translates a list of fragment strings to a list of integers.

        Parameters:
        sentence (list of strings) [sequence_length]: The sequence of fragments that make up the molecule.
        Example: seq[:-1] = ['<SOS>', '*c1ccc(C)cc1', '*[NH+]1CCCCCC1', '*CC(*)*', '*N*', '*C(*)=O', '*c1ccc(F)cc1']
                  seq[1:] = ['*c1ccc(C)cc1', '*[NH+]1CCCCCC1', '*CC(*)*', '*N*', '*C(*)=O', '*c1ccc(F)cc1', '<EOS>']

        Returns:
        sentence_translated (list of integers) [sequence_length]: The sequence of fragments that make up the molecule
        translated to integers. This would inclued the SOS and EOS tokens.
        Example: vocab.translate(seq[:-1]) = [0, 575, 363, 158, 437, 81, 529]
                  vocab.translate(seq[1:]) = [575, 363, 158, 437, 81, 529, 2]
        """
        sentence_translated = []
        for word in sentence:
            
            # Check if the word is in the vocabulary and is not a special token
            if word not in self.TOKEN_IDS:
                if isinstance(word, np.integer):
                    sentence_translated.append(self.translate_int(word))
                elif isinstance(word, str):
                    sentence_translated.append(self.translate_str(word))
            
            # if the word is an EOS token, break the loop
            elif word == self.EOS:
                break
        return sentence_translated