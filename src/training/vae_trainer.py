import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from tensorboardX import SummaryWriter
from torch.autograd import Variable
from models.vae_model import VAEModel, Loss
from data.dataset import MoleculeFragmentsDataset

from torch.nn.utils import clip_grad_norm_
from utils.training_utils import load_checkpoint, save_checkpoint, get_optimizer, get_scheduler, dump

class VAETrainer:
    """
    This class is responsible for training and validating a VAE model.
    """
    @classmethod
    def load(cls, config, vocab, last):
        trainer = VAETrainer(config, vocab)
        epoch = load_checkpoint(trainer, last=last)
        return trainer, epoch
    
    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

        # initialise the model, optimizer, scheduler and loss class
        self.model = VAEModel(config, vocab)
        self.optimizer = get_optimizer(config, self.model)
        self.scheduler = get_scheduler(config, self.optimizer)
        self.criterion = Loss(config, vocab, pad=vocab.PAD)

        # load the model to the GPU if specified
        if self.config.get('use_gpu') is True:
            self.model = self.model.cuda()
        
        # initialise the beta values for KL annealing
        self.beta_KL = []

        # initialise the loss lists
        self.pred_logp_loss = None
        if self.config.get('pred_logp') is True:
            self.pred_logp_loss = []
        if self.config.get('pred_sas') is True:
            self.pred_sas_loss = []
        self.losses = []
        self.CE_loss = []
        self.KL_loss = []
        self.best_loss = float('inf')

    def train(self, loader, start_epoch)->None:
        """
        This method trains the model for a specified number of epochs.
        """
        # get the number of epochs
        num_epochs = self.config.get('num_epochs')

        # get the logger
        logger = TensorBoardLogger(self.config)

        # Get counts of each fragments and calculate penalty weights
        dataset = MoleculeFragmentsDataset(self.config)
        penalty_weights = self.get_fragment_penalty_weights(dataset)

        # initialise the beta values for KL annealing
        beta = [0, 0, 0, 0, 0, 0.002, 0.006, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1]
        #beta = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 0.5]
        print('beta:', beta)
        self.beta_list = beta

        for epoch in range(start_epoch, start_epoch + num_epochs):
            start = time.time()

            # train the model for one epoch
            if self.config.get('pred_logp') or self.config.get('pred_sas'):
                epoch_loss, CE_epoch_loss, KL_epoch_loss, logp_loss, sas_loss = self._train_epoch(epoch, loader, penalty_weights, beta)
            else:
                epoch_loss, CE_epoch_loss, KL_epoch_loss = self._train_epoch(epoch, loader, penalty_weights, beta)
            
            # update the loss lists
            if self.config.get('pred_logp'):
                self.pred_logp_loss.append(logp_loss)
            if self.config.get('pred_sas'):
                self.pred_sas_loss.append(sas_loss)
            self.losses.append(epoch_loss)
            self.CE_loss.append(CE_epoch_loss)
            self.KL_loss.append(KL_epoch_loss)

            # log the loss values to TensorBoard
            logger.log('loss', epoch_loss, epoch)
            logger.log('CE_loss', CE_epoch_loss, epoch)
            logger.log('KL_loss', KL_epoch_loss, epoch)
            if self.config.get('pred_logp'):
                logger.log('logp_loss', logp_loss, epoch)
            if self.config.get('pred_sas'):
                logger.log('sas_loss', sas_loss, epoch)

            # save the checkpoint
            save_checkpoint(self, epoch, filename="last.pt")
            
            # save the best loss checkpoint
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                save_checkpoint(self, epoch, filename=f'best_loss.pt')

            # print the epoch loss
            self.log_epoch(start, epoch, epoch_loss)
        dump(self.config, self.losses, self.CE_loss, self.KL_loss, self.pred_logp_loss, self.pred_sas_loss, self.beta_list)
    
    def _train_epoch(self, epoch, loader, penalty_weights, beta)->tuple:
        """
        This method trains the model for one epoch.

        Parameters:
        epoch (int): The current epoch
        loader (DataLoader): The data loader
        penalty_weights (list): The penalty weights for the KL loss
        beta (list): The beta values for KL annealing

        Returns:
        tuple:
            - float: The average loss for the epoch
            - float: The average CE loss for the epoch
            - float: The average KL loss for the epoch
            - float: The average predicted logP loss for the epoch (if specified)
            - float: The average predicted SAS loss for the epoch (if specified)
        """
        # set the model to train mode
        self.model.train()

        # load the dataset
        dataset = MoleculeFragmentsDataset(self.config)

        # initialise the loss values
        epoch_pred_logp_loss = 0
        epoch_pred_sas_loss = 0
        epoch_loss = 0
        epoch_CE_loss = 0
        epoch_KL_loss = 0

        # initialise the logp and sas labels
        labels_logp = None
        labels_sas = None

        # apply the learning rate scheduler
        if epoch > 0 and self.config.get('use_scheduler'):
            self.scheduler.step()
        
        for idx, (src, tgt, lengths, data_index, tgt_str) in enumerate(loader):
            # zero the gradients
            self.optimizer.zero_grad()

            # convert the input sequences to variables
            src, tgt = Variable(src), Variable(tgt)

            # move the input sequences to the GPU if specified  
            if self.config.get('use_gpu'):
                src = src.cuda()
                tgt = tgt.cuda()

            # predict the molecular properties if specified
            if self.config.get('pred_logp') or self.config.get('pred_sas'):
                # get the target string list
                tgt_str_lst = [self.vocab.translate(target_i) for target_i in tgt.cpu().detach().numpy()]

                # join the target string list and separate by space to compare to data
                tgt_str_lst_join = [" ".join(self.vocab.translate(target_i)) for target_i in tgt.cpu().detach().numpy()]
                output, mu, sigma, z, pred_logp, pred_sas = self.model(src, lengths)

                # get the corrected labels for logp and sas for the randomised batched data
                molecules = dataset.data.iloc[list(data_index)]
                data_index_correct = [molecules[molecules['fragments'] == tgt_str_lst_join_i].index.values[0] for tgt_str_lst_join_i in tgt_str_lst_join]
                molecules_correct = dataset.data.iloc[data_index_correct]
                labels_logp = torch.tensor(molecules_correct.logP.values)
                labels_sas = torch.tensor(molecules_correct.SAS.values)

                # move the labels to the GPU if specified
                if self.config.get('use_gpu'):
                    labels_logp = labels_logp.cuda()
                    labels_sas = labels_sas.cuda()
            else:
                output, mu, sigma, z = self.model(src, lengths)
            
            # calculate the loss
            loss, CE_loss, KL_loss, logp_loss, sas_loss = self.criterion(output, tgt, mu, sigma, pred_logp, labels_logp, pred_sas, labels_sas, epoch, penalty_weights, beta)
            loss.backward()

            # clip the gradients to prevent exploding gradients
            clip_grad_norm_(self.model.parameters(), self.config.get('clip_norm'))
            
            # update the loss values
            epoch_loss += loss.item()
            epoch_CE_loss += CE_loss.item()
            epoch_KL_loss += KL_loss.item()
            if self.config.get('pred_logp'):
                epoch_pred_logp_loss += logp_loss.item()
            if self.config.get('pred_sas'):
                epoch_pred_sas_loss += sas_loss.item()

            # update the model parameters
            self.optimizer.step()

            # print the loss values every 1000 iterations
            if idx == 0 or idx % 1000 == 0:
                self.print_loss(self.config, epoch, CE_loss, KL_loss, pred_logp, labels_logp, logp_loss, pred_sas, labels_sas, sas_loss, data_index, beta)
        
        if self.config.get('pred_logp') or self.config.get('pred_sas'):
            return epoch_loss / len(loader), epoch_CE_loss / len(loader), epoch_KL_loss / len(loader), epoch_pred_logp_loss / len(loader), epoch_pred_sas_loss / len(loader)
        else:
            return epoch_loss / len(loader), epoch_CE_loss / len(loader), epoch_KL_loss / len(loader)

    def get_fragment_penalty_weights(self, dataset)->np.ndarray:
        """
        Function that returns the penalty weights for the KL loss

        Parameters:
        dataset (FragmentDataset): The dataset object
        """
        fragment_list = []
        print('Calculating fragment penalty weights...')
        for frag in tqdm(dataset.data.fragments):
            fragment_list.extend(frag.split())
        fragment_counts = pd.Series(fragment_list).value_counts()
        penalty = np.sum(np.log(fragment_counts + 1)) / np.log(fragment_counts + 1)
        penalty_weights = penalty / np.linalg.norm(penalty) * 50
        return penalty_weights
    
    def print_loss(config, epoch, CE_loss, KL_loss, pred_logp, labels_logp, logp_loss, pred_sas, labels_sas, sas_loss, data_index, beta)->None:
        """
        This method prints the loss values for the current epoch.

        Parameters:
        config (Config): The configuration object
        epoch (int): The current epoch
        CE_loss (float): The cross entropy loss
        KL_loss (float): The KL loss
        pred_logp (torch.Tensor): The predicted logP values
        labels_logp (torch.Tensor): The labels for the logP values
        logp_loss (float): The logP loss
        pred_sas (torch.Tensor): The predicted SAS values
        labels_sas (torch.Tensor): The labels for the SAS values
        sas_loss (float): The SAS loss
        data_index (list): The indices of the data
        beta (list): The beta values for KL annealing
        """
        print(f"Epoch: {epoch}, beta: {beta[epoch]:.2f}")
        print(f"index: {data_index}")
        if config.get('pred_logp'):
            formatted_logp = [f"{value:.4f}" for value in pred_logp.flatten()]
            formatted_logp_labels = [f"{value:.4f}" for value in labels_logp.tolist()]
            logp_pred_str = f"pred logp: {formatted_logp}" if pred_logp is not None else "pred logp: None"
            logp_label_str = f"labels logp: {formatted_logp_labels}"
            logp_loss_str = f"logP Loss: {logp_loss.item():.4f}"
            print(logp_pred_str)
            print(logp_label_str)
            print(logp_loss_str)
        if config.get('pred_sas'):
            formatted_sas = [f"{value:.4f}" for value in pred_sas.flatten()]
            formatted_sas_labels = [f"{value:.4f}" for value in labels_sas.tolist()]
            sas_pred_str = f"pred sas: {formatted_sas}" if pred_sas is not None else "pred sas: None"
            sas_label_str = f"labels sas: {formatted_sas_labels}"
            sas_loss_str = f"SAS Loss: {sas_loss.item():.4f}"
            print(sas_pred_str)
            print(sas_label_str)
            print(sas_loss_str)
        CE_loss_str = f"{CE_loss.item():.4f}"
        KL_loss_str = f"{KL_loss.item():.4f}"
        print(f"CE Loss: {CE_loss_str}, KL Loss: {KL_loss_str}")

class TensorBoardLogger:
    """
    This class is responsible for logging the loss values to TensorBoard.
    """

    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(self.config.path('tensorboard').as_posix())
        self.config.write_summary(self.writer)

    def log(self, tag, value, epoch):
        """
        This method logs the loss values to TensorBoard.

        Parameters:
        tag (str): The tag for the loss value
        value (float): The loss value
        epoch (int): The current epoch
        """
        self.writer.add_scalar(tag, value, epoch)

    def close(self):
        """
        This method closes the TensorBoard writer.
        """
        self.writer.close()