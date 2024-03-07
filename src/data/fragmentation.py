import numpy as np

from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles, BRICS

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

def break_into_fragments(mol, smi):
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