from src.data.fragmentation import break_into_fragments_defragmo
from src.utils.mol_utils import mol_from_smiles

def test_fragmentation_defragmo():
    smi = "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"
    assert break_into_fragments_defragmo(smi, min_length=0) == ('CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1', '*C(C)(C)C *C(=O)Cc1coc2ccc(*)cc12 *N* c1(*)ccccc1F', 4)
    

if __name__=="__main__":
    test_fragmentation_defragmo()
    print("all tests passed!")

   