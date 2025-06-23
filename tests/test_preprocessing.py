from src.data.preprocess import pre_process_zinc, canonicalize_and_drop
from io import StringIO
import pandas as pd

def test_read_and_clean_zinc_dataset():

    input_json_str = """
        [
            {
                "smiles":"CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1\n",
                "logP":5.0506,
                "qed":0.7020122328,
                "SAS":2.0840945721
            },
            {
                "smiles":"C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1\n",
                "logP":3.1137,
                "qed":0.9289754881,
                "SAS":3.4320038193
            },
            {
                "smiles":"N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1\n",
                "logP":4.96778,
                "qed":0.5996817382,
                "SAS":2.4706326078
            },
            {
                "smiles":"CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1\n",
                "logP":4.00022,
                "qed":0.690944353,
                "SAS":2.8227533112
            },
            {
                "smiles":"N#CC1=C(SCC(=O)Nc2cccc(Cl)c2)N=C([O-])[C@H](C#N)C12CCCCC2\n",
                "logP":3.60956,
                "qed":0.7890271546,
                "SAS":4.0351821383
            }
        ]
    """
    expected_output_json = """
        [
            {
                "smiles":"CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",
                "logP":5.0506,
                "qed":0.7020122328,
                "SAS":2.0840945721
            },
            {
                "smiles":"CC1CC(C)CC(Nc2cncc(-c3nncn3C)c2)C1",
                "logP":3.1137,
                "qed":0.9289754881,
                "SAS":3.4320038193
            },
            {
                "smiles":"N#Cc1ccc(-c2ccc(OC(C(=O)N3CCCC3)c3ccccc3)cc2)cc1",
                "logP":4.96778,
                "qed":0.5996817382,
                "SAS":2.4706326078
            },
            {
                "smiles":"CCOC(=O)C1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1",
                "logP":4.00022,
                "qed":0.690944353,
                "SAS":2.8227533112
            },
            {
                "smiles":"N#CC1=C(SCC(=O)Nc2cccc(Cl)c2)N=C([O-])C(C#N)C12CCCCC2",
                "logP":3.60956,
                "qed":0.7890271546,
                "SAS":4.0351821383
            }
        ]
    """
    dataset = pd.read_json(StringIO(input_json_str), orient = "records")
    dataset = pre_process_zinc(dataset)
    dataset = canonicalize_and_drop(dataset)
    dataset_expected = pd.read_json(StringIO(expected_output_json), orient="records")
    pd.testing.assert_frame_equal(dataset, dataset_expected)

def main():
    pass

if __name__=="__main__":
    main()
