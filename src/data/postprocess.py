import time
import numpy as np
import pandas as pd

from utils.mol_utils import mol_from_smiles

def mask_valid_molecules(smiles):
    valid_mask = []

    for smi in smiles:
        try:
            mol = mol_from_smiles(smi)
            valid_mask.append(mol is not None)
        except Exception:
            valid_mask.append(False)

    return np.array(valid_mask)


def mask_novel_molecules(smiles, data_smiles):
    novel_mask = []

    for smi in smiles:
        novel_mask.append(smi not in data_smiles)

    return np.array(novel_mask)


def mask_unique_molecules(smiles):
    uniques, unique_mask = set(), []

    for smi in smiles:
        unique_mask.append(smi not in uniques)
        uniques.add(smi)

    return np.array(unique_mask)

def score_samples(samples, dataset, calc=True)->tuple:
    """
    Function to score the samples based on validity, novelty, and uniqueness.

    Parameters:
    samples (pd.DataFrame or list): the samples to score
    dataset (pd.DataFrame): the dataset of molecules
    calc (bool): whether to calculate the scores

    Returns:
    Tuple:
    - list: the scores of validity, novelty, and uniqueness
    """
    def ratio(mask):
        total = mask.shape[0]
        if total == 0:
            return 0.0
        return mask.sum() / total

    if isinstance(samples, pd.DataFrame):
        smiles = samples.smiles.tolist()
    elif isinstance(samples, list):
        smiles = [s[0] for s in samples]
    data_smiles = dataset.smiles.tolist()

    valid_mask = mask_valid_molecules(smiles)
    novel_mask = mask_novel_molecules(smiles, data_smiles)
    unique_mask = mask_unique_molecules(smiles)

    scores = []
    if calc:
        start = time.time()
        print("Start scoring...")
        validity_score = ratio(valid_mask)
        novelty_score = ratio(novel_mask[valid_mask])
        uniqueness_score = ratio(unique_mask[valid_mask])

        print(f"valid: {validity_score} - "
              f"novel: {novelty_score} - "
              f"unique: {uniqueness_score}")

        scores = [validity_score, novelty_score, uniqueness_score]
        end = time.time() - start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
        print(f'Done. Time elapsed: {elapsed}.')

    return scores