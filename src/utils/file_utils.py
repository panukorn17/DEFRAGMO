import pandas as pd
import pickle as pkl


def load_pickle(path):
    """
    This function loads a pickle file from the path.

    Parameters:
    - path (string): the directory of the path to load the file from, including the file name.

    Returns:
    .pkl file
    """
    return pkl.load(open(path, "rb"))


def save_pickle(obj, path):
    """
    This function saves a the object as a pickle file.

    Parameters:
    - obj: the object to be saved as a pickle file.
    - path (string): the directory of the path to save the object in.

    """
    
    pkl.dump(obj, open(path, "wb"))

def load_data():
    """
    This function loads the data from the source file.

    Returns:
    data (pandas.DataFrame) [number_of_molecules, features]: the source of the full dataset
    """
    data = pd.read_csv('DATA/ZINC/PROCESSED/train.smi').reset_index(drop=True)
    return data