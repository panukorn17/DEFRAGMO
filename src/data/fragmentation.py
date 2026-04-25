from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

from utils.mol_utils import mol_from_smiles, mol_to_smiles, mols_to_smiles, root_smiles

# Suppress all RDKit warnings and errors
RDLogger.DisableLog("rdApp.*")
###################### PODDA's Fragmentation functions ######################
dummy = Chem.MolFromSmiles("[*]")


def strip_dummy_atoms(mol):
    hydrogen = mol_from_smiles("[H]")
    mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
    mol = Chem.RemoveHs(mols[0])
    return mol


def count_dummies(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count


def join_molecules(molA, molB):
    marked, neigh = None, None
    for atom in molA.GetAtoms():
        if atom.GetAtomicNum() == 0:
            marked = atom.GetIdx()
            neigh = atom.GetNeighbors()[0]
            break
    neigh = 0 if neigh is None else neigh.GetIdx()

    if marked is not None:
        ed = Chem.EditableMol(molA)
        ed.RemoveAtom(marked)
        molA = ed.GetMol()

    joined = Chem.ReplaceSubstructs(molB, dummy, molA, replacementConnectionPoint=neigh, useChirality=False)[0]

    # Chem.Kekulize(joined)
    return joined


def reconstruct(frags, reverse=False):
    if len(frags) == 1:
        return strip_dummy_atoms(frags[0]), frags

    try:
        if count_dummies(frags[0]) != 1:
            print("yes 1")
            # print(mol_to_smiles(frags[1]))
            # print(count_dummies(frags[1]))
            return None, None

        if count_dummies(frags[-1]) != 1:
            print("yes 2")
            return None, None

        for frag in frags[1:-1]:
            if count_dummies(frag) != 2:
                print("yes 3")
                return None, None

        mol = join_molecules(frags[0], frags[1])
        for i, frag in enumerate(frags[2:]):
            print(i, mol_to_smiles(frag), mol_to_smiles(mol))
            mol = join_molecules(mol, frag)
            print(i, mol_to_smiles(mol))

        # see if there are kekulization/valence errors
        mol_to_smiles(mol)

        return mol, frags
    except Exception:
        return None, None


def break_on_bond(mol, bond, min_length=3):
    if mol.GetNumAtoms() - bond <= min_length:
        return [mol]

    broken = Chem.FragmentOnBonds(mol, bondIndices=[bond], dummyLabels=[(0, 0)])

    res = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=False)

    return res


def get_size(frag):
    dummies = count_dummies(frag)
    total_atoms = frag.GetNumAtoms()
    real_atoms = total_atoms - dummies
    return real_atoms


def get_dummy_count(frag):
    return count_dummies(frag)


def fragment_iterative(mol, min_length=1):

    bond_data = list(BRICS.FindBRICSBonds(mol))

    try:
        idxs, labs = zip(*bond_data)
    except Exception:
        return []

    bonds = []
    for a1, a2 in idxs:
        bond = mol.GetBondBetweenAtoms(a1, a2)
        bonds.append(bond.GetIdx())

    order = np.argsort(bonds).tolist()
    bonds = [bonds[i] for i in order]

    frags, temp = [], deepcopy(mol)
    for bond in bonds:
        res = break_on_bond(temp, bond)

        if len(res) == 1:
            frags.append(temp)
            break

        head, tail = res
        if get_size(head) < min_length or get_size(tail) < min_length:
            continue

        frags.append(head)
        temp = deepcopy(tail)

    return frags


def break_into_fragments_podda(mol, smi):
    frags = fragment_iterative(mol)

    if len(frags) == 0:
        return smi, np.nan, 0

    if len(frags) == 1:
        return smi, smi, 1

    rec, frags = reconstruct(frags)
    if rec and mol_to_smiles(rec) == smi:
        fragments = mols_to_smiles(frags)
        return smi, " ".join(fragments), len(frags)

    return smi, np.nan, 0


###################### DEFRAGMO's Fragmentation functions ######################


@dataclass
class FragContext:
    "Fragmentation context for each molecule"

    smi_orig: str
    smi: str
    frags: list = field(default_factory=list)
    counter: int = 0
    max_brics_bonds: int = 0
    min_length: int = 0
    verbose: int = 0


def replace_last(s: str, old: str, new: str) -> str:
    """
    Function to replace the last occuring dummy label with a fragment.

    Parameters:
    s (str): the string (fragment) to which the dummy label * is to be replaced with another fragment
    old (str): the string from the fragment s to be replaced
    new (str): the string to replace the "old" string in the fragment s

    Returns:
    str: the original string s with the replacemnt
    """
    s_reversed = s[::-1]
    old_reversed = old[::-1]
    new_reversed = new[::-1]

    # Replace the first occurrence in the reversed string
    s_reversed = s_reversed.replace(old_reversed, new_reversed, 1)

    # Reverse the string back to original order
    return s_reversed[::-1]


def check_reconstruction(
    frags: list[str],
    frag_1: str,
    frag_2: str,
    orig_smi,
) -> bool:
    """Function to test whether the original molecule has been reconstructed."""
    try:
        frags_test = frags.copy()
        frags_test.append(frag_1)
        frags_test.append(frag_2)
        frag_2_re = frags_test[-1]
        for i in range(len(frags_test) - 1):
            frag_1_re = frags_test[-1 * i - 2]
            recomb = replace_last(frag_2_re, "*", frag_1_re.replace("*", "", 1))
            recomb_canon = root_smiles(recomb, rootedAtAtom=1)
            frag_2_re = recomb_canon
        orig_smi_canon = root_smiles(orig_smi, rootedAtAtom=1)
        if recomb_canon == orig_smi_canon:
            return True
        else:
            return False
    except Exception:
        return False


def check_bond_no(
    bonds: list,
    frags: list,
    max_brics_bonds: int,
    smi: str,
    verbose: int = 0,
) -> tuple:
    """
    This function checks if the molecule has less bonds than the limit of BRIC bonds.

    Parameters:
    bonds (list): the list of BRIC bonds
    smi (str): the smiles string of the molecule
    frags (list): the list of fragments
    frag_list_len (int): the length of the fragment list

    Returns:
    tuple: a tuple containing the fragment list and a boolean value to indicate whether fragmentation is complete
        - frags (list): the list of fragments
        - fragComplete (bool): a boolean value to indicate whether fragmentation is complete
    """
    if len(bonds) <= max_brics_bonds:
        if verbose == 1:
            print("Final Fragment: ", smi)
        frags.append(root_smiles(smi, rootedAtAtom=1))
        fragComplete = True
        return frags, fragComplete
    else:
        fragComplete = False
        return frags, fragComplete


def get_head_tail(mol, bond):
    broken = Chem.FragmentOnBonds(mol, bondIndices=[bond], dummyLabels=[(0, 0)])
    head, tail = Chem.GetMolFrags(broken, asMols=True)
    head_bric_bond_no = len(list(BRICS.FindBRICSBonds(head)))
    tail_bric_bond_no = len(list(BRICS.FindBRICSBonds(tail)))
    return head, tail, head_bric_bond_no, tail_bric_bond_no


def get_bond_idxs(mol) -> list[int]:
    """Get sorted BRICS bond indices from a molecule."""
    bonds = list(BRICS.FindBRICSBonds(mol))
    idxs, labs = list(zip(*bonds))
    bond_idxs = []
    for a1, a2 in idxs:
        bond = mol.GetBondBetweenAtoms(a1, a2)
        bond_idxs.append(bond.GetIdx())
    order = np.argsort(bond_idxs).tolist()
    return [bond_idxs[i] for i in order]


def try_assign_fragment(ctx, primary, remainder, bond_idxs, bond, label="head"):
    "Try to assign primary as a valid fragment, recurse on remainder"
    primary_smi = mol_to_smiles(primary)
    remainder_smi = mol_to_smiles(remainder, rootedAtAtom=1)
    if (
        check_reconstruction(ctx.frags, primary_smi, remainder_smi, ctx.smi_orig)
        and get_size(primary) >= ctx.min_length
    ):
        # Condition 1: reconstruction succeeds and fragment is large enough - accept and recurse
        if ctx.verbose:
            print(f"{label} fragment: ", primary_smi)
            print(f"Recurse : {'tail' if label == 'head' else label}", remainder_smi)
        ctx.frags.append(primary_smi)
        sub = FragContext(ctx.smi_orig, remainder_smi, ctx.frags, ctx.counter, 0, ctx.min_length, ctx.verbose)
        fragComplete = fragment_recursive(sub)
        if fragComplete:
            return ctx.frags, fragComplete
    elif len(bond_idxs) == 1 and get_size(mol_from_smiles(ctx.smi)) >= ctx.min_length:
        # Condition 2: one bond to try but reconstruction failed - the whole molecule is a fragment
        if ctx.verbose:
            print("Final Fragment: ", ctx.smi)
        ctx.frags.append(root_smiles(ctx.smi, rootedAtAtom=1))
        return ctx.frags, True
    elif bond == bond_idxs[-1]:
        # Condition 3: tried all bonds at this recursion level but none worked,
        # increase max_brix_bond tolerance by 1 and recurse
        sub = FragContext(
            ctx.smi_orig,
            root_smiles(ctx.smi, rootedAtAtom=1),
            ctx.frags,
            ctx.counter,
            ctx.max_brics_bonds + 1,
            ctx.min_length,
            ctx.verbose,
        )
        fragComplete = fragment_recursive(sub)
        if fragComplete:
            return ctx.frags, fragComplete
    return ctx.frags, False


def try_fragment_bond(ctx, head, tail, head_bric_bond_no, tail_bric_bond_no, bond_idxs, bond):
    "Try to assign head or tail as the primary fragment for this bond"
    if head_bric_bond_no <= ctx.max_brics_bonds:
        return try_assign_fragment(ctx, head, tail, bond_idxs, bond, label="head")
    if tail_bric_bond_no <= ctx.max_brics_bonds:
        return try_assign_fragment(ctx, tail, head, bond_idxs, bond, label="tail")
    return ctx.frags, False


def fragment_recursive(ctx: FragContext) -> list[str]:
    "Recursively fragment a molecule using the DEFRAGMO method"
    try:
        ctx.counter += 1
        mol = mol_from_smiles(ctx.smi)
        bonds = list(BRICS.FindBRICSBonds(mol))
        ctx.frags, fragComplete = check_bond_no(bonds, ctx.frags, ctx.max_brics_bonds, ctx.smi, ctx.verbose)
        if fragComplete:
            return ctx.frags
        bond_idxs = get_bond_idxs(mol)
        for bond in bond_idxs:
            head, tail, head_bric_bond_no, tail_bric_bond_no = get_head_tail(mol, bond)
            ctx.frags, fragComplete = try_fragment_bond(
                ctx, head, tail, head_bric_bond_no, tail_bric_bond_no, bond_idxs, bond
            )
            if fragComplete:
                return ctx.frags
    except Exception as e:
        print(f"Exception: {e}")


def break_into_fragments_defragmo(smi, min_length=0, verbose=0):
    RDLogger.DisableLog("rdApp.*")
    "Fragment a molecule using the DEFRAGMO breadth-first BRICS fragmentation method"
    ctx = FragContext(
        smi_orig=smi, smi=smi, frags=[], counter=0, max_brics_bonds=0, min_length=min_length, verbose=verbose
    )
    frags = fragment_recursive(ctx)
    return smi, " ".join(frags), len(frags)


if __name__ == "__main__":
    smi = "CCCN(CCc1cccc(-c2ccccc2)c1)C(=O)C1OC(C(=O)O)=CC(N)C1NC(C)=O"
    print(break_into_fragments_defragmo(smi, min_length=0, verbose=1))
