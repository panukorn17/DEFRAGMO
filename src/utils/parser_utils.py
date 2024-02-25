import argparse

def setup_parser()-> argparse.ArgumentParser:
    """
    Sets up the command line argument parser

    Returns:
    parser: The command line argument parser
    """
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # create the parser for the "train" command
    parser_train = subparsers.add_parser('train', help='Train the vae model')
    parser_train.add_argument('--data_name', type=str, help='The name of the model')
    parser_train.add_argument('--use_gpu', action='store_true', help='Whether to use the GPU')
    parser_train.add_argument('--batch_size', type=int, help='The batch size')
    parser_train.add_argument('--num_workers', type=int, help='The number of workers')
    parser_train.add_argument('--num_epochs', type=int, help='The number of epochs')
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
    return parser