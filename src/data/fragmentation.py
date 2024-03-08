import numpy as np

from rdkit import Chem
from copy import deepcopy
from rdkit.Chem import MolToSmiles, MolFromSmiles, BRICS
from utils.mol_utils import mol_to_smiles, mols_to_smiles, mol_from_smiles

###################### PODDA's Fragmentation functions ######################
dummy = Chem.MolFromSmiles('[*]')

def strip_dummy_atoms(mol):
    hydrogen = mol_from_smiles('[H]')
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

    joined = Chem.ReplaceSubstructs(
        molB, dummy, molA,
        replacementConnectionPoint=neigh,
        useChirality=False)[0]

    #Chem.Kekulize(joined)
    return joined

def reconstruct(frags, reverse=False):
    if len(frags) == 1:
        return strip_dummy_atoms(frags[0]), frags

    try:
        if count_dummies(frags[0]) != 1:
            print("yes 1")
            #print(mol_to_smiles(frags[1]))
            #print(count_dummies(frags[1]))
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

    broken = Chem.FragmentOnBonds(
        mol, bondIndices=[bond],
        dummyLabels=[(0, 0)])

    res = Chem.GetMolFrags(
        broken, asMols=True, sanitizeFrags=False)

    return res

def get_size(frag):
    dummies = count_dummies(frag)
    total_atoms = frag.GetNumAtoms()
    real_atoms = total_atoms - dummies
    return real_atoms

def fragment_iterative(mol, min_length=3):

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
def replace_last(s, old, new):
    s_reversed = s[::-1]
    old_reversed = old[::-1]
    new_reversed = new[::-1]

    # Replace the first occurrence in the reversed string
    s_reversed = s_reversed.replace(old_reversed, new_reversed, 1)

    # Reverse the string back to original order
    return s_reversed[::-1]

def check_reconstruction(frags, frag_1, frag_2, orig_smi):
    try:
        #print("Reconstructing...")
        frags_test = frags.copy()
        frags_test.append(frag_1)
        frags_test.append(frag_2)
        frag_2_re = frags_test[-1]
        for i in range(len(frags_test)-1):
            frag_1_re = frags_test[-1*i-2]
            recomb = replace_last(frag_2_re, "*", frag_1_re.replace("*", "",1))
            recomb_canon = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(recomb)),rootedAtAtom = 1)
            frag_2_re = recomb_canon
        orig_smi_canon = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(orig_smi)),rootedAtAtom = 1)
        if recomb_canon == orig_smi_canon:
            #print("Reconstruction successful")
            #print("Original Smiles:", orig_smi, "Fragment 1:" , frag_1, "Fragment 2: ", frag_2, "Reconstruction: ", recomb_canon)
            return True
        else:
            #print("Reconstruction failed")
            #print("True Smiles:", smi, "Fragment 1:" , frag_1, "Fragment 2: ", frag_2, "Reconstruction: ", recomb_canon)
            return False
    except:
        #print("Reconstruction failed")
        #print("True Smiles:", smi, "Fragment 1:" , frag_1, "Fragment 2: ", frag_2, "Reconstruction: ", recomb_canon)
        return False

def fragment_recursive(mol_smi_orig, mol_smi, frags, counter, frag_list_len):
    fragComplete = False
    try:
        counter += 1
        mol = MolFromSmiles(mol_smi)
        bonds = list(BRICS.FindBRICSBonds(mol))
        if len(bonds) <= frag_list_len:
            frags.append(MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1))
            #rint("Final Fragment: ", mol_smi, "Number of BRIC bonds: ", len(bonds))
            fragComplete = True
            return fragComplete

        idxs, labs = list(zip(*bonds))

        bond_idxs = []
        for a1, a2 in idxs:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            bond_idxs.append(bond.GetIdx())

        order = np.argsort(bond_idxs).tolist()
        bond_idxs = [bond_idxs[i] for i in order]
        for bond in bond_idxs:
            broken = Chem.FragmentOnBonds(mol,
                                        bondIndices=[bond],
                                        dummyLabels=[(0, 0)])
            head, tail = Chem.GetMolFrags(broken, asMols=True)
            head_bric_bond_no = len(list(BRICS.FindBRICSBonds(head)))
            tail_bric_bond_no = len(list(BRICS.FindBRICSBonds(tail)))
            if head_bric_bond_no <= frag_list_len:
                head_smi = Chem.CanonSmiles(MolToSmiles(head))
                tail_smi = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(MolToSmiles(tail))), rootedAtAtom=1)
                if check_reconstruction(frags, head_smi, tail_smi, mol_smi_orig):
                    frags.append(head_smi)
                    #print("Recursed: ", mol_smi, "Bond: ", bond, "Terminal: ", head_smi, "Number of BRIC bonds: ", head_bric_bond_no, "Recurse: ", tail_smi)
                    fragComplete = fragment_recursive(mol_smi_orig, tail_smi, frags, counter, frag_list_len = 0)  
                    if fragComplete:
                        return frags
                elif len(bond_idxs) == 1:
                    frags.append(MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1))
                    #print("Final Fragment: ", mol_smi, "Number of BRIC bonds: ", len(bonds))
                    fragComplete = True
                    return frags
                elif bond == bond_idxs[-1]:
                    fragComplete = fragment_recursive(mol_smi_orig, MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1), frags, counter, frag_list_len + 1)
                    if fragComplete:
                        return frags
            elif tail_bric_bond_no <= frag_list_len:
                tail_smi = Chem.CanonSmiles(MolToSmiles(tail))
                head_smi = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(MolToSmiles(head))), rootedAtAtom=1)
                if check_reconstruction(frags, tail_smi, head_smi, mol_smi_orig):
                    frags.append(tail_smi)
                    #print("Recursed: ", mol_smi, "Bond: ", bond,  "Terminal: ", tail_smi, "Number of BRIC bonds: ", tail_bric_bond_no, "Recurse: ", head_smi)
                    fragComplete = fragment_recursive(mol_smi_orig, head_smi, frags, counter, frag_list_len = 0)  
                    if fragComplete:
                        return frags
                elif len(bond_idxs) == 1:
                    frags.append(MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1))
                    #print("Final Fragment: ", mol_smi, "Number of BRIC bonds: ", len(bonds))
                    fragComplete = True
                    return frags
                elif bond == bond_idxs[-1]:
                    fragComplete = fragment_recursive(mol_smi_orig, MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1), frags, counter, frag_list_len + 1)
                    if fragComplete:
                        return frags
    except Exception:
        pass

def break_into_fragments_defragmo(mol, smi):
    #frags = fragment_iterative(mol)
    frags = []
    fragment_recursive(smi, smi, frags, 0, 0)

    if len(frags) == 0:
        return smi, np.nan, 0

    if len(frags) == 1:
        return smi, smi, 1
    """
    rec, frags = reconstruct(frags)
    if rec and mol_to_smiles(rec) == smi:
        fragments = mols_to_smiles(frags)
        return smi, " ".join(fragments), len(frags)

    return smi, np.nan, 0
    """
    return smi, " ".join(frags), len(frags)