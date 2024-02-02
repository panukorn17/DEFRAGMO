from rdkit import Chem

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
    return Chem.MolFromSmiles(Chem.Canonicalize(smiles))

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