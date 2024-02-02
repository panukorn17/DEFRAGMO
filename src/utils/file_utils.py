import pandas as pd

def load_data():
    """
    This function loads the data from the source file.

    Returns:
    data: the source of the full dataset
    """
    data = pd.read_csv('DATA/ZINC/PROCESSED/train.smi').reset_index(drop=True)
    return data