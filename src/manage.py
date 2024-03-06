from data.dataset import MoleculeFragmentsDataset
from training.vae_trainer import VAETrainer
from utils.config import Config

from utils.parser_utils import setup_parser
from utils.config import get_data_info

def preprocess(data_name:str)->None:
    """
    This function is responsible for preprocessing the data which is used to train the VAE model.

    Parameters:
    data_name (str): The name of the dataset e.g. ZINC.
    """
    data_info = get_data_info(data_name)
    dataset = clean_dataset(data_info)
    dataset = add_atom_counts(dataset, data_info)
    dataset = add_bond_counts(dataset, data_info)
    dataset = add_ring_counts(dataset, data_info)
    

def train_vae(config):
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

if __name__ == '__main__':
    debug = False
    if debug:
        # parse the arguments and call the function
        parser = setup_parser()
        simulated_args = [
                    'train',
                    '--data_name', 'ZINC',
                    '--num_epochs', '15',
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
            
        # parse the arguments and create a dictionary of the arguments
        args = vars(parser.parse_args(simulated_args))
    else: 
        # parse the arguments and call the function
        parser = setup_parser()
        args = vars(parser.parse_args())
        
    # get the command and remove it from the dictionary
    command = args.pop('command')

    if command == 'preprocess':
        data_name = args.pop('data_name')
        preprocess(data_name)
    if command == 'train':
        config = Config(**args)
        train_vae(config)