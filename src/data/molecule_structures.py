import copy

from rdkit import Chem
from collections import OrderedDict
from rdkit.Chem import Crippen, QED
from data.sascorer.sascorer import calculateScore


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
    bond_count (OrderedDict): an ordered dict of the bonds in the bond_list with the count of each atom.
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

def count_rings(molecule:Chem.rdchem.Mol, ring_list:list)->OrderedDict:
    """
    This function is responsible for counting the rings from the ring_list that are in the molecule.

    Parameters:
    molecule (Chem.rdchem.Mol): the molecule object
    ring_list (list): the list of rings to count

    Returns:
    ring_count (OrderedDict): an ordered dict of the rings in the ring_list with the count of each atom.
    """
    # the ring_sizes dict correspond to the following lists in the info file for each dataset
    # "rings": ["Tri", "Quad", "Pent", "Hex"]
    # "ring_sizes": [3, 4, 5, 6]
    ring_sizes = {i: r for (i, r) in zip(range(3, 7), ring_list)}

    # initialise the ordered dict for ring_count
    ring_count = OrderedDict(zip(ring_list, [0]*len(ring_list)))
    if molecule:
        # get the ring info
        ring_info = Chem.GetSymmSSSR(molecule)

        for ring in ring_info:
            ring_length = len(list(ring))
            if ring_length in ring_sizes:
                ring_name = ring_sizes[ring_length]
                ring_count[ring_name] += 1
    return ring_count


def logp(mol):
    return Crippen.MolLogP(mol) if mol else None


def mr(mol):
    return Crippen.MolMR(mol) if mol else None


def qed(mol):
    return QED.qed(mol) if mol else None


def sas(mol):
    return calculateScore(mol) if mol else None