import pandas as pd

from datetime import datetime
from data.dataset import MoleculeFragmentsDataset
from training.vae_trainer import VAETrainer
from models.vae_sampler import Sampler
from utils.config import Config
from tqdm import tqdm

from utils.file_utils import load_data
from utils.parser_utils import setup_parser
from utils.config import get_data_info, DATA_DIR
from utils.mol_utils import mols_from_smiles
from utils.plot_utils import plot_paper_figures
from data.preprocess import read_and_clean_dataset, add_atom_counts, add_bond_counts, add_ring_counts, add_property, add_fragments_defragmo, add_fragments_podda, save_dataset
from data.postprocess import score_samples

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
    save_dataset(dataset, data_info)
    

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

def sample_model(config:Config):
    # get date and time
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dataset = MoleculeFragmentsDataset(config)
    vocab = dataset.set_vocab()
    load_last = config.get('load_last')
    trainer, _ = VAETrainer.load(config, vocab, last=load_last)
    var_const = config.get('sample_constant')
    sampler = Sampler(config, vocab, trainer.model, var_const)
    # get training set mean
    train_mean, train_std = sampler.get_train_mean_std(dataset)
    #seed = config.get('sampling_seed') if config.get('reproduce') else None
    samples_tup = sampler.sample(config.get('num_samples'), train_mean, train_std)
    # update list of tuples (smiles, frags, num_frags) to df
    samples_list = [(smi, frags, num_frags) for smi, frags, num_frags in samples_tup]
    samples_df = pd.DataFrame(samples_list, columns=['smiles', 'fragments', 'n_fragments'])
    samples_df.to_csv(config.path('results') / (date_time + f"_std_{int(var_const)}_samples.smi"), index=False)
    dataset = load_data(config, data_type="test")
    scores = score_samples(samples_df, dataset)
    print(f"Scores: {scores}")

def plot_samples(config:Config, sample_name:str):
    run_dir = config.path('run')
    run_name = config.get('run_name')
    plot_paper_figures(run_dir, run_name, sample_name)
    
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
        """simulated_args = [
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
                ]"""
        
        # simulated arguments for sampling
        """simulated_args = [
                'sample',
                '--run_dir', 'src/runs/2024-07-04-12-49-55-ZINC',
                '--sampler_method', 'greedy',
                '--load_last',
                '--num_samples', '100',
                '--max_len', '10'
                ]"""
        
        # simulated arguments for plotting
        simulated_args = [
                'plot',
                '--run_dir', 'src/runs/2024-07-05-09-39-45-ZINC',
                '--sample_name', '2024-07-05-10-52-18_std_25_samples',
                ]

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
        elif command == 'plot':
            run_dir = args.pop('run_dir')
            sample_name = args.pop('sample_name')
            config = Config.load(run_dir, **args)
            plot_samples(config, sample_name)
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
        elif command == 'train':
            config = Config(**args)
            train_vae(config)
        elif command == 'sample':
            run_dir = args.pop('run_dir')
            args.update({'use_gpu': False})
            config = Config.load(run_dir, **args)
            sample_model(config)
        elif command == 'plot':
            run_dir = args.pop('run_dir')
            sample_name = args.pop('sample_name')
            config = Config.load(run_dir, **args)
            plot_samples(config, sample_name)