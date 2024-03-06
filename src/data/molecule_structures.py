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
    atom_count = OrderedDict()
    for atom in atom_list:
        if atom not in atom_count:
            atom_count[atom] = 0
        else:
            atom_count[atom] += 1
    
    