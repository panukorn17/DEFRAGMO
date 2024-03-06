import pandas as pd
from utils.config import DATA_DIR
from utils.mol_utils import canonicalize

def read_and_clean_dataset(info):
    raw_path = DATA_DIR / info['name'] / 'raw'

    dataset = pd.read_csv(
        raw_path / info['filename'],
        index_col=info['index_col'])

    if info['drop'] != []:
        dataset = dataset.drop(info['drop'], axis=1)

    if info['name'] == 'ZINC':
        dataset = dataset.replace(r'\n', '', regex=True)

    if info['name'] == 'GDB17':
        dataset = dataset.sample(n=info['random_sample'])
        dataset.columns = ['smiles']

    if info['name'] == 'PCBA':
        cols = dataset.columns.str.startswith('PCBA')
        dataset = dataset.loc[:, ~cols]
        dataset = dataset.drop_duplicates()
        dataset = dataset[~dataset.smiles.str.contains("\.")]

    if info['name'] == 'QM9':
        correct_smiles = pd.read_csv(raw_path / 'gdb9_smiles_correct.csv')
        dataset.smiles = correct_smiles.smiles
        dataset = dataset.sample(frac=1, random_state=42)

    smiles = dataset.smiles.tolist()
    dataset.smiles = [canonicalize(smi, clear_stereo=True) for smi in smiles]
    dataset = dataset[dataset.smiles.notnull()].reset_index(drop=True)

    return dataset