from data.dataset import MoleculeFragmentsDataset
from training.vae_trainer import VAETrainer
from utils.parser_utils import setup_parser
from utils.config import Config

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
    """dataset = MoleculeFragmentsDataset()
    dataset.set_vocab()

    loader = dataset.get_loader()
    for idx, (src_tensor, tgt_tensor, src_lengths, tgt_lengths) in enumerate(loader):
        print(src_tensor, tgt_tensor)
        """
    # parse the arguments and call the function
    parser = setup_parser()
    simulated_args = [
                'train',
                '--data_name', 'ZINC',
                '--num_epochs', '1'
            ]
        
    # parse the arguments and create a dictionary of the arguments
    args = vars(parser.parse_args(simulated_args))

    # get the command and remove it from the dictionary
    command = args.pop('command')

    if command == 'train':
        config = Config(**args)
        train_vae(config)