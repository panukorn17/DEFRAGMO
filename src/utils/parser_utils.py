import argparse

def setup_parser()-> argparse.ArgumentParser:
    """
    Sets up the command line argument parser

    Returns:
    parser: The command line argument parser
    """
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # create the parser for the "preprocess" command
    parser_fragment = subparsers.add_parser('preprocess', help='Preprocess the dataset for training')
    parser_fragment.add_argument(
        '--data_name', type=str, default = 'ZINC', 
        choices=['ZINC'], 
        help='The name of the dataset')
    parser_fragment.add_argument(
        '--method', type=str, default='DEFRAGMO',
        choices=['DEFRAGMO','PODDA'],
        help='the method to fragment the dataset'
    )

    # create the parser for the "train" command
    parser_train = subparsers.add_parser('train', help='Train the vae model')
    parser_train.add_argument(
        '--data_name', type=str, default = 'ZINC', 
        choices=['ZINC'], 
        help='The name of the dataset')
    parser_train.add_argument(
        '--use_gpu', action='store_true', 
        help='Whether to use the GPU')
    parser_train.add_argument(
        '--batch_size', type=int, 
        help='The batch size')
    parser_train.add_argument(
        '--num_workers', type=int, 
        help='The number of workers')
    parser_train.add_argument(
        '--num_epochs', type=int, 
        help='The number of epochs')
    parser_train.add_argument(
        '--pooling', default = None,
        choices=['mean', 'max', 'sum_fingerprints', None],
        help='pooling method for the encoder, if any')
    parser_train.add_argument(
        '--pred_logp', default = False,
        action='store_true',
        help='predict logP')
    parser_train.add_argument(
        '--pred_sas', default = False,
        action='store_true',
        help='predict SAS')
    parser_train.add_argument(
        '--embed_size',
        default=64, type=int,
        help='size of the embedding layer')
    parser_train.add_argument(
        '--hidden_size',
        default=64, type=int,
        help='number of recurrent neurons per layer')
    parser_train.add_argument(
        '--hidden_layers',
        default=2, type=int,
        help='number of recurrent layers')
    parser_train.add_argument(
        '--dropout',
        default=0.3, type=float,
        help='dropout for the recurrent layers')
    parser_train.add_argument(
        '--latent_size',
        default=100, type=int,
        help='size of the VAE latent space')
    parser_train.add_argument(
        '--lr', dest='optim_lr',
        default=0.00001, type=float,
        help='learning rate')
    parser_train.add_argument(
        '--no_scheduler', dest='use_scheduler',
        action='store_false',
        help="don't use learning rate scheduler")
    parser_train.add_argument(
        '--step_size', dest='sched_step_size',
        default=2, type=int,
        help='annealing step size for the scheduler')
    parser_train.add_argument(
        '--gamma', dest='sched_gamma',
        default=0.9, type=float,
        help='annealing rate for the scheduler')
    parser_train.add_argument(
        '--clip_norm',
        default=5.0, type=float,
        help='threshold to clip the gradient norm')
    parser_train.add_argument(
        '--embed_method',
        default='mol2vec', type=str,
        choices=['mol2vec', 'skipgram'],
        help='window for word2vec embedding')    
    parser_train.add_argument(
        '--embed_window',
        default=3, type=int,
        help='window for word2vec embedding')
    parser_train.add_argument(
        '--use_mask', dest='use_mask',
        action='store_true',
        help="use mask for low-frequency fragments")
    parser_train.add_argument(
        '--mask_freq',
        type=int, default=2,
        help="masking frequency")
    parser_train.add_argument(
        '--beta',
        type=float, 
        nargs='+', default=None,
        help="KL Annealing list which must match the number of epochs")
    
    # create the parser for the "sample" command
    parser_sample = subparsers.add_parser('sample', help='Sample from the vae model')
    parser_sample.add_argument(
        '--run_dir', metavar="FOLDER",
        help="directory of the run in the format src/runs/<name_of_run>.")
    parser_sample.add_argument(
        '--load_last', action="store_true",
        help='load last model instead of best')
    parser_sample.add_argument(
        '--num_samples',
        default=1000, type=int,
        help='number of samples to draw from the model')
    parser_sample.add_argument(
        '--max_len',
        default=10, type=int,
        help='maximum length of the sampled sequence')
    parser_sample.add_argument(
        '--sampler_method', 
        default='greedy', type=str,
        choices=['greedy', 'sample_first', 'sample_all'],
        help='sampling method')
    parser_sample.add_argument(
        '--temperature',
        default=1.0, type=float,
        help='sampling temperature')
    parser_sample.add_argument(
        '--sample_repeat', default= None,
        choices=['unique_all', 'unique_adjacent', None],
        help='how to keep or resample tokens.')
    parser_sample.add_argument(
        '--sample_constant', 
        default= 1, type=float, 
        help='constant used to multiply to the sampling variance.')

    # create the parser for the "plot" command
    parser_plot = subparsers.add_parser('plot', help='Plot the sample properties from a model')
    parser_plot.add_argument(
        '--run_dir', metavar="FOLDER",
        help="directory of the run in the format src/runs/<name_of_run>.")
    parser_plot.add_argument(
        '--sample_name', default=None, type=str,
        help='The name of the file containing the sampled molecules.')
    return parser