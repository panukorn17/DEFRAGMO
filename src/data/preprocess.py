import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from utils.config import DATA_DIR
from utils.mol_utils import canonicalize, mols_from_smiles
from data.molecule_structures import count_atoms, count_bonds, count_rings, qed, sas, logp, mr
from data.fragmentation import break_into_fragments
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
    print("Canonicalizing Molecules...")
    dataset.smiles = [canonicalize(smi, clear_stereo=True) for smi in tqdm(smiles)]
    # drop all null rows
    dataset = dataset[dataset.smiles.notnull()].reset_index(drop=True)

    return dataset

def add_atom_counts(dataset:pd.DataFrame, mols:list, info:dict)->pd.DataFrame:
    """
    This function adds the count of atoms in each molecule to the dataset dataframe.

    Parameters:
    dataset (pd.DataFrame): dataset of the molecules
    mols (list of Chem.rdchem.Mol): list of molecules in Chem.rdchem.Mol format
    info (dict): the information dictionary of the dataset

    Returns:
    pd.DataFrame: the dataframe of molecules with the atom counts concatenated.
    """
    print("Counting Atoms...")
    counts = [count_atoms(mol, info['atoms']) for mol in tqdm(mols)]
    return pd.concat([dataset, pd.DataFrame(counts)], axis=1, sort=False)


def add_bond_counts(dataset:pd.DataFrame, mols:list, info:dict)->pd.DataFrame:
    """
    This function adds the count of bonds in each molecule to the dataset dataframe.

    Parameters:
    dataset (pd.DataFrame): dataset of the molecules
    mols (list of Chem.rdchem.Mol): list of molecules in Chem.rdchem.Mol format
    info (dict): the information dictionary of the dataset

    Returns:
    pd.DataFrame: the dataframe of molecules with the bond counts concatenated.
    """
    print("Counting Bonds...")
    counts = [count_bonds(mol, info['bonds']) for mol in tqdm(mols)]
    return pd.concat([dataset, pd.DataFrame(counts)], axis=1, sort=False)


def add_ring_counts(dataset:pd.DataFrame, mols:list, info:dict)->pd.DataFrame:
    """
    This function adds the count of rings in each molecule to the dataset dataframe.

    Parameters:
    dataset (pd.DataFrame): dataset of the molecules
    mols (list of Chem.rdchem.Mol): list of molecules in Chem.rdchem.Mol format
    info (dict): the information dictionary of the dataset

    Returns:
    pd.DataFrame: the dataframe of molecules with the ring counts concatenated.
    """
    print("Counting Rings...")
    counts = [count_rings(mol, info['rings']) for mol in tqdm(mols)]
    return pd.concat([dataset, pd.DataFrame(counts)], axis=1, sort=False)

def add_property(dataset:pd.DataFrame, mols:list, prop_name:str)->pd.DataFrame:
    """
    This function adds the property of each molecule to the dataset dataframe.
    
    Parameters:
    dataset (pd.DataFrame): dataset of the molecules
    mols (list of Chem.rdchem.Mol): list of molecules in Chem.rdchem.Mol format
    prop_name (str): the name of the property to add to the dataframe

    Returns:
    pd.DataFrame: the dataframe of molecules property names
    """
    fn = {"qed": qed, "SAS": sas, "logP": logp, "mr": mr}[prop_name]
    prop = [fn(mol) for mol in mols]
    new_data = pd.DataFrame(prop, columns=[prop_name])
    return pd.concat([dataset, new_data], axis=1, sort=False)

def add_fragments(dataset:pd.DataFrame, mols:list, smiles:list)->pd.DataFrame:
    """
    This function adds fragments the modlecules in the dataset dataframe.

    Parameters:
    dataset (pd.DataFrame): dataset of the molecules
    mols (list of Chem.rdchem.Mol): list of molecules in Chem.rdchem.Mol format
    smiles (list of str): list of smiles strings

    Returns:
    dataset: the dataframe of molecules with fragments
    """
    results = [break_into_fragments(m, s) for m, s in zip(mols, smiles)]
    smiles, fragments, lengths = zip(*results)
    dataset["smiles"] = smiles
    dataset["fragments"] = fragments
    dataset["n_fragments"] = lengths
    
    return dataset