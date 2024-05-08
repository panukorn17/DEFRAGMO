from datetime import datetime
from data.dataset import MoleculeFragmentsDataset
from training.vae_trainer import VAETrainer
from models.vae_sampler import Sampler
from utils.config import Config
from tqdm import tqdm

from utils.parser_utils import setup_parser
from utils.config import get_data_info, DATA_DIR
from utils.mol_utils import mols_from_smiles
from data.preprocess import read_and_clean_dataset, add_atom_counts, add_bond_counts, add_ring_counts, add_property, add_fragments_defragmo, add_fragments_podda

def preprocess(data_name:str, method:str)->None:
    """
    This function is responsible for preprocessing the data which is used to train the VAE model.

    Parameters:
    data_name (str): The name of the dataset e.g. ZINC.
    method (str): The name of the method to fragment the molecules
    """
    data_info = get_data_info(data_name)
    dataset = read_and_clean_dataset(data_info)
    smiles = dataset.smiles.tolist()
    mols = mols_from_smiles(smiles)
    dataset = add_atom_counts(dataset, mols, data_info)
    dataset = add_bond_counts(dataset, mols, data_info)
    dataset = add_ring_counts(dataset, mols, data_info)

    print("Adding properties ...")
    for prop in tqdm(data_info['properties']):
        if prop not in dataset.columns:
            dataset = add_property(dataset, mols, prop)
    if method == 'DEFRAGMO':
        dataset = add_fragments_defragmo(dataset, mols, smiles)
    elif method == 'PODDA':
        dataset = add_fragments_podda(dataset, mols, smiles)
    dataset = dataset[["smiles","fragments","n_fragments","C","F","N","O","Other","SINGLE","DOUBLE","TRIPLE","Tri","Quad","Pent","Hex","logP","mr","qed","SAS"]]
    dataset.to_csv((DATA_DIR / data_name / f'processed/processed_{method}.smi').as_posix(), index=False)
    

def train_vae(config:Config)->None:
    """
    This function is responsible for training the VAE model.

    Parameters:
    config (Config): The configuration of the run.
    """
    # create an instance of the dataset
    dataset = MoleculeFragmentsDataset(config)
    vocab = dataset.set_vocab()
    trainer = VAETrainer(config, vocab)
    trainer.train(dataset.get_loader(), start_epoch=0)

def sample_model(config):
    # get date and time
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dataset = MoleculeFragmentsDataset(config)
    vocab = dataset.set_vocab()
    load_last = config.get('load_last')
    trainer, _ = VAETrainer.load(config, vocab, last=load_last)
    sampler = Sampler(config, vocab, trainer.model)
    #seed = config.get('sampling_seed') if config.get('reproduce') else None
    samples = sampler.sample(config.get('num_samples'))
    with open(config.path('results') / (date_time + "_samples.smi"), 'w') as f:
        for sample in samples:
            sample_str = ' '.join(map(str, sample))
            f.write(sample_str + '\n')
    
if __name__ == '__main__':
    debug = False
    if debug:
        # parse the arguments and call the function
        parser = setup_parser()

        # simulated arguments for preprocessing
        """simulated_args = [
                    'preprocess',
                    '--data_name', 'ZINC',
                    '--method', 'PODDA'
                ]"""
        #python  src/manage.py preprocess --data_name ZINC --method DEFRAGMO
        
        # simulated arguments for model training
        simulated_args = [
                    'train',
                    '--data_name', 'ZINC',
                    '--num_epochs', '5',
                    '--batch_size', '16',
                    '--embed_size', '100',
                    '--hidden_layers', '2',
                    '--hidden_size', '64',
                    '--latent_size', '32',
                    '--pooling', 'sum_fingerprints',
                    '--use_gpu',
                    '--pred_logp',
                    '--pred_sas'
                ]
        
        # simulated arguments for sampling
        """
        simulated_args = [
                'sample',
                '--run_dir', 'src/runs/2024-05-03-11-05-40-ZINC',
                '--sampler_method', 'sample_all',
                '--load_last'
                ]
        """
        # parse the arguments and create a dictionary of the arguments
        args = vars(parser.parse_args(simulated_args))
        # get the command and remove it from the dictionary
        command = args.pop('command')
        if command == 'preprocess':
            data_name = args.pop('data_name')
            method = args.pop('method')
            preprocess(data_name, method)
        elif command == 'train':
            config = Config(**args)
            train_vae(config)
        elif command == 'sample':
            run_dir = args.pop('run_dir')
            args.update({'use_gpu': False})
            config = Config.load(run_dir, **args)
            sample_model(config)
    else: 
        # parse the arguments and call the function
        parser = setup_parser()
        args = vars(parser.parse_args())
        
    # get the command and remove it from the dictionary
    command = args.pop('command')

    if command == 'preprocess':
        data_name = args.pop('data_name')
        method = args.pop('method')
        preprocess(data_name, method)
    if command == 'train':
        config = Config(**args)
        train_vae(config)
    if command == 'sample':
        run_dir = args.pop('run_dir')
        args.update({'use_gpu': False})
        config = Config.load(run_dir, **args)
        sample_model(config)