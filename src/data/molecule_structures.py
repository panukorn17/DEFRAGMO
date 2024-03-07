import copy

from rdkit import Chem
from collections import OrderedDict

def count_atoms(molecule:Chem.rdchem.Mol, atom_list:list)->OrderedDict:
    """
    This function is responsible for counting the atoms from the atom_list that are in the molecule.

    Parameters:
    molecule (Chem.rdchem.Mol): the molecule object
    atom_list: the list of atoms to count

    Returns:
    atom_count (OrderedDict): an ordered dict of the atoms in the atom_list with the count of each atom.
    """
    # initialise the ordered dict for atom_count
    atom_count = OrderedDict(zip(atom_list, [0]*len(atom_list)))
    if molecule:
        # get the atom objects from the molecule
        mol_atoms = molecule.GetAtoms()
        for atom in mol_atoms:
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in atom_count:
                atom_count["Other"] = 0
            else:
                atom_count[atom_symbol] += 1
    return atom_count
    
def count_bonds(molecule:Chem.rdchem.Mol, bond_list:list)->OrderedDict:
    """
    This function is responsible for counting the atoms from the atom_list that are in the molecule.

    Parameters:
    molecule (Chem.rdchem.Mol): the molecule object
    bond_list (list): the list of bonds to count

    Returns:
    bond_count (OrderedDict): an ordered dict of the bonds in the atom_list with the count of each atom.
    """
    # initialise the ordered dict for bond_count
    bond_count = OrderedDict(zip(bond_list, [0]*len(bond_list)))
    if molecule:
        molecule = copy.deepcopy(molecule)
        Chem.Kekulize(molecule, clearAromaticFlags=True)
        
        # get the bond objects from the molecule
        mol_bonds = molecule.GetBonds()
        for bond in mol_bonds:
            bond_count[str(bond.GetBondType())] += 1
    return bond_count