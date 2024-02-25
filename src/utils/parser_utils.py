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
    parser_train.add_argument('--use_gpu', type=bool, help='Whether to use the GPU')
    parser_train.add_argument('--batch_size', type=int, help='The batch size')
    parser_train.add_argument('--num_workers', type=int, help='The number of workers')
    parser_train.add_argument('--num_epochs', type=int, help='The number of epochs')

    return parser