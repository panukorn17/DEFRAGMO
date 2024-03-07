import pandas as pd

from rdkit import Chem
from data.molecule_structures import count_atoms

def mol_to_smiles(mol):
    """
    This function converts a molecule to a SMILES string.

    Parameters:
    mol: the molecule to convert

    Returns:
    str: The SMILES string of the molecule
    """
    return Chem.Canonicalize(Chem.MolToSmiles(mol))

def mol_from_smiles(smiles):
    """
    This function converts a SMILES string to a molecule.

    Parameters:
    smiles: the SMILES string to convert

    Returns:
    Mol: The molecule of the SMILES string
    """
    return Chem.MolFromSmiles(Chem.CanonSmiles(smiles))

def mols_to_smiles(mols):
    """
    This function converts a list of molecules to a list of SMILES strings.

    Parameters:
    mols: the list of molecules to convert

    Returns:
    list: The list of SMILES strings of the molecules
    """
    return [mol_to_smiles(mol) for mol in mols]

def mols_from_smiles(smiles):
    """
    This function converts a list of SMILES strings to a list of molecules.

    Parameters:
    smiles: the list of SMILES strings to convert

    Returns:
    list: The list of molecules of the SMILES strings
    """
    return [mol_from_smiles(smile) for smile in smiles]

def canonicalize(smiles:str, clear_stereo=False):
    """
    This function returns the canonicalised smiles representation and has the option
    to clear stereochemistry i.e. remove the @@.

    Parameters:
    smiles (str): the smiles representation of the molecule.
    clear_stereo (bool): boolean variable to clear stereochemistry of a molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if clear_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def add_atom_counts(dataset:pd.DataFrame, info:dict)->pd.DataFrame:
    """
    This function adds the count of atoms in each molecule to the dataset dataframe.

    Parameters:
    dataset (pd.DataFrame): dataset of the molecules
    info (dict): the information dictionary of the dataset
    """
    smiles = dataset.smiles.tolist()
    mols = mols_from_smiles(smiles)
    counts = [count_atoms(mol, info['atoms']) for mol in mols]
    return pd.concat([dataset, pd.DataFrame(counts)], axis=1, sort=False)


def add_bond_counts(dataset, info, n_jobs):
    return _add_counts(dataset, count_bonds, info['bonds'], n_jobs)


def add_ring_counts(dataset, info, n_jobs):
    return _add_counts(dataset, count_rings, info['rings'], n_jobs)
