import os

from pathlib import Path
from datetime import datetime

from file_utils import load_pickle, save_pickle, save_json

PROJECT_DIR = Path('./src')
DATA_DIR = PROJECT_DIR / 'data'
RUNS_DIR = PROJECT_DIR / 'runs'
DEFAULT_EXPT_PARAMS = {
    # General
    'title': 'Molecular VAE',
    'description': 'An RNN based Molecular VAE',
    'log_dir': RUNS_DIR.as_posix(),
    'random_seed': 42,
    'use_gpu': False,

    # Data
    'batch_size': 16,
    'shuffle': True,

    # Model
    'embed_size': 100,
    'hidden_size': 64,
    'hidden_layers': 2,
    'latent_size': 32,
    'dropout': 0.3,
    'pooling': None,
    'pred_logp': False,
    'pred_sas': False,

    # Training
    'num_epochs': 10,
    'optim_lr': 0.001,
    'use_scheduler': True,
    'sched_step_size': 2,
    'sched_gamma': 0.1,
    'clip_norm': 5.0,
    }

class Config:
    """
    This class is responsible for handling the configuration of the project.
    """

    # define the class attributes
    FILENAME = 'config.pkl'
    JSON_FILENAME = 'config.json'

    @classmethod
    def load(cls, run_dir, **run_params):
        """
        Loads the configuration from the run directory.

        Parameters:
        run_dir (str): The directory of the run.
        run_params (dict): The parameters of the run.

        Returns:
        config (Config): The configuration object.
        """
        # create the path variable to the config file
        path = Path(run_dir) / 'config' / cls.FILENAME

        # load the config file
        config = load_pickle(path)

        # update the config with the run parameters
        config.update(**run_params)
        
        return config
    
    # define the instance attributes
    def __init__(self, dataset, **run_params):
        """
        The constructor which initialises the configuration object with the default parameters.

        Parameters:
        params (dict): The parameters of the configuration.
        """
        # Get the run info
        run_name, start_date_time = get_run_info(run_params.get('data_name', 'default'))
        data_path = DATA_DIR / dataset / 'PROCESSED'
        params = DEFAULT_EXPT_PARAMS.copy()

        # add parameters to the params attribute
        params.update({
            'run_name': run_name,
            'start_date_time': start_date_time,
            'data_path': data_path.as_posix(),
            'dataset': dataset
        })

        # create the directory for the run
        paths_dict = create_run_dir(RUNS_DIR, run_name, data_path)

        # update the params with the run parameters
        for key, value in run_params.items():
            if key in params:
                params[key] = value

        # set the params attribute
        self.params = params

        # set the paths attribute
        self.paths = paths_dict

        # save the config to the run directory
        self.save()
    
    def save(self):
        """
        Saves the configuration to the run directory.
        """
        # save the config to the run directory as a pickle file
        path = Path(self.paths['config']) / self.FILENAME
        save_pickle(self.params, path)

        # save the config to the run directory as a json file
        json_path = Path(self.paths['config']) / self.JSON_FILENAME
        save_json(self.params, json_path)
    
    def update(self, **new_params):
        """
        Updates the configuration with the new parameters.

        Parameters:
        new_params (dict): The new parameters of the configuration.
        """
        # update the params with the run parameters
        for key, value in new_params.items():
            if key in self.params:
                self.params[key] = value



def get_run_info(data_name: str) -> tuple:
    """
    Gets the run info.

    Parameters:
    name (str): The name of the data to train the model.

    Returns:
    tuple:
        - run_name (str): The name of the run.
        - start_date_time (str): The start time
    """
    start_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_name = f'{start_date_time}-{data_name}'
    return run_name, start_date_time

def create_run_dir(root: str, run_name: str, data_path: str)->dict:
    """
    Creates the run directory.
    
    Parameters:
    root (str): The root directory.
    run_name (str): The name of the run.
    data_path (str): The path to the data.

    Returns:
    dict: A dictionary containing the paths of the run.
    """
    paths_dict = {'data': data_path}

    # create the src/runs directory
    os.makedirs(root, exist_ok=True)

    # create the src/runs/run_name directory
    run_dir = root / run_name
    paths_dict['run'] = run_dir
    os.makedirs(run_dir, exist_ok=True)

    # create the config directory
    config_dir = run_dir / 'config'
    paths_dict['config'] = config_dir
    os.makedirs(config_dir, exist_ok=True)

    # create the tensorboard directory
    log_dir = run_dir / 'logs'
    paths_dict['log'] = log_dir
    os.makedirs(log_dir, exist_ok=True)

    # create the results directory
    results_dir = run_dir / 'results'
    paths_dict['results'] = results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    return paths_dict
