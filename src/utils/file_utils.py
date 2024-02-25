import pandas as pd
import pickle as pkl
import json


def load_pickle(path):
    """
    This function loads a pickle file from the path.

    Parameters:
    path (string): the directory of the path to load the file from, including the file name.

    Returns:
    .pkl file
    """
    return pkl.load(open(path, "rb"))


def save_pickle(obj, path):
    """
    This function saves a the object as a pickle file.

    Parameters:
    obj: the object to be saved as a pickle file.
    path (string): the directory of the path to save the object in.
    """
    
    pkl.dump(obj, open(path, "wb"))

def save_json(obj, path):
    """
    This function saves a the object as a json file.

    Parameters:
    obj: the object to be saved as a json file.
    path (string): the directory of the path to save the object in.
    """
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)

def load_data(config, data_type):
    """
    This function loads the data from the source file.

    Parameters:
    config (Config): the configuration of the runs.
    data_type (str): selection between training set or test set.

    Returns:
    data (pandas.DataFrame) [number_of_molecules, features]: the source of the full dataset
    """
    assert data_type in ['train', 'test']
    data_path = config.path('data') / f'{data_type}.smi'
    data = pd.read_csv(data_path).reset_index(drop=True)
    return data