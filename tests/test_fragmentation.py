from src.data.fragmentation import break_into_fragments_defragmo
from src.utils.mol_utils import mol_from_smiles
from rdkit import Chem


def test_fragmentation_defragmo():
    smi = "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"
    mol = mol_from_smiles(smi)
    assert break_into_fragments_defragmo(mol,smi) == ('CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1', '*C(C)(C)C *C(=O)Cc1coc2ccc(*)cc12 *N* c1(*)ccccc1F', 4)
    

if __name__=="__main__":
    test_fragmentation_defragmo()

   