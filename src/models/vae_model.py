import numpy as np
import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

class VAEModel(nn.Module):
    """
    This class represents the variational autoencoder model. It is responsible for the training and evaluation of the model.
    This class inherits from the nn.Module class in PyTorch.
    """

    def __init__(self, config, vocab):
        """
        The constructor for the VAEModel class.

        Parameters:
        config (Config): the configuration of the model
        vocab (Vocabulary): the vocabulary of the model
        """
        super().__init__()

        # set the configuration, vocabulary, parameters and embeddings
        self.config = config
        self.vocab = vocab
        self.input_size = vocab.get_size()
        self.embed_size = config.get('embed_size')
        self.hidden_size = config.get('hidden_size')
        self.hidden_layers = config.get('hidden_layers')
        self.latent_size = config.get('latent_size')
        self.dropout = config.get('dropout')
        self.use_gpu = config.get('use_gpu')

        embeddings = self.load_embeddings()
        self.embedder = nn.Embedding.from_pretrained(embeddings)

        # set the encoder
        self.encoder = Encoder(
            config=self.config,
            input_size=self.input_size,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            latent_size=self.latent_size,
            dropout=self.dropout,
            use_gpu=self.use_gpu
            )
        
        # set the decoder
        self.decoder = Decoder(
            config=self.config,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            latent_size=self.latent_size,
            output_size=self.input_size,
            dropout=self.dropout,
            use_gpu=self.use_gpu
            )
        
        # set the linear layers
        self.latent2hidden = Latent2Hidden(
            latent_size=self.latent_size,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            use_gpu=self.use_gpu
            )
        
        # set the mlp
        if self.config.get('pred_logp') or self.config.get('pred_sas'):
            self.mlp = MLP(
                config=self.config,
                latent_size=self.latent_size,
                hidden_size_mlp=self.hidden_size,
                hidden_layers_mlp=self.hidden_layers,
                dropout=self.dropout)
            
    def forward(self, inputs, lengths):
        """
        This method performs a forward pass through the model.

        Parameters:
        inputs (torch.Tensor): the input data
        lengths (list): the lengths of the input sequences

        Returns:
        output (torch.Tensor): the output of the model
        """
        # embeddings is of shape (batch_size, seq_len, embed_size)
        embeddings = self.embedder(inputs)

        if self.config.get('pooling') == 'sum_fingerprints':
            # embeddings1 is of shape (batch_size, embed_size)
            embeddings1 = self.sum_fingerprints(inputs, self.embed_size)
        else:
            # embeddings is of shape (batch_size, seq_len, embed_size)
            embeddings1 = F.dropout(embeddings, p=self.dropout, training=self.training)
        # z, mu, sigma are all of shape (batch_size, latent_size)
        z, mu, sigma = self.encoder(inputs, embeddings1, lengths)
        if self.config.get('pred_logp') or self.config.get('pred_sas'):
            mlp_outputs = self.mlp(z)
            # logp, sas are both of shape (batch_size, 1)
            logp = mlp_outputs[0] if self.config.get('pred_logp') else None
            sas = mlp_outputs[1] if self.config.get('pred_sas') else None
        # state is of shape (hidden_layers, batch_size, hidden_size)
        state = self.latent2hidden(z)
        embeddings2 = F.dropout(embeddings, p=self.dropout, training=self.training)
        
        # output is of shape (batch_size, seq_len, output_size)
        # state is of shape (hidden_layers, batch_size, hidden_size)
        output, state = self.decoder(embeddings2, state, lengths)
        if self.config.get('pred_logp') or self.config.get('pred_sas'):
            return output, mu, sigma, z, logp, sas
        else:
            return output, mu, sigma, z
        
    def load_embeddings(self):
        """
        This method loads the pre-trained embeddings.

        Returns:
        embeddings (torch.Tensor): the pre-trained embeddings
        """
        file_name = f'emb_{self.embed_size}.dat'
        path = self.config.path('config') / file_name
        embeddings = np.loadtxt(path, delimiter=',')
        return torch.from_numpy(embeddings).float()

class Encoder(nn.Module):
    """
    This class is the encoder of the VAE model. It is responsible for the encoding of the input sequence to the latent space.
    This class inherits from the nn.Module class in PyTorch.
    """
    def __init__(self, config, input_size, embed_size, hidden_size, hidden_layers, latent_size, dropout, use_gpu):
        """
        The constructor for the Encoder class.

        Parameters:
        config (Config): the configuration of the model
        input_size (int): the size of the input
        embed_size (int): the size of the embeddings
        hidden_size (int): the size of the hidden layers
        hidden_layers (int): the number of hidden layers
        latent_size (int): the size of the latent space
        dropout (float): the dropout rate
        use_gpu (bool): whether to use the GPU
        """
        super().__init__()

        self.config = config
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.latent_size = latent_size
        self.dropout = dropout
        self.use_gpu = use_gpu

        self.rnn = nn.GRU(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            dropout=self.dropout,
            batch_first=True
        )

        # adapt the input size depending on the pooling method
        if self.config.get('pooling') in ['max', 'mean', 'sum']:
            input_size = self.hidden_size
        elif self.config.get('pooling') == 'sum_fingerprints':
            input_size = self.embed_size
        else:
            input_size = self.hidden_size * self.hidden_layers

        # set the linear layers
        self.rnn2mean = nn.Linear(
            in_features=input_size, 
            out_features=self.latent_size
            )
        
        self.rnn2logv = nn.Linear(
            in_features=input_size, 
            out_features=self.latent_size
            )

    def forward(self, inputs, embeddings, lengths):
        batch_size = inputs.size(0)
        if self.config.get('pooling') == 'sum_fingerprints':
            # mean is of shape (batch_size, latent_size)
            mean = self.rnn2mean(embeddings)

            # logv is of shape (batch_size, latent_size)
            logv = self.rnn2logv(embeddings)
        else:
            # Let GRU initialize to zeros
            # packed is of shape (sum(lengths), embed_size)
            # lengths is a list of lengths for each sequence in the batch
            packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=True)
            
            # the packed_output is of shape (batch_size, seq_len, hidden_size)
            # the state is of shape (hidden_layers, batch_size, hidden_size)
            packed_output, state = self.rnn(packed)
            
            # output is of shape (batch_size, seq_len, hidden_size)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            if self.config.get('pooling') == 'max':
                pooled, _ = torch.max(output, dim=1)
            elif self.config.get('pooling') == 'mean':
                pooled = torch.mean(output, dim=1)
            elif self.config.get('pooling') == 'sum':
                pooled = torch.sum(output, dim=1)
            else:
                state = state.view(batch_size, self.hidden_size * self.hidden_layers)
            
            # mean is of shape (batch_size, latent_size)
            mean = self.rnn2mean(pooled if self.config.get('pooling') != None else state)
            
            # logv is of shape (batch_size, latent_size)
            logv = self.rnn2logv(pooled if self.config.get('pooling') != None else state)
        # std is of shape (batch_size, latent_size)
        std = torch.exp(0.5 * logv)
        
        # z is of shape (batch_size, latent_size)
        z = torch.randn_like(mean)
        
        # latent_sample, mean, and std is of shape (batch_size, latent_size)
        latent_sample = z * std + mean
        
        return latent_sample, mean, std


class Decoder(nn.Module):
    """
    This class is the decoder of the VAE model. It is responsible for the decoding of the latent space to the data space.
    This class inherits from the nn.Module class in PyTorch.
    """
    
    def __init__(self, config, embed_size, hidden_size, hidden_layers, latent_size, output_size, dropout, use_gpu):
        """
        The constructor for the Decoder class.

        Parameters:
        config (Config): the configuration of the run
        embed_size (int): the size of the embeddings
        hidden_size (int): the size of the hidden layers
        hidden_layers (int): the number of hidden layers
        latent_size (int): the size of the latent space
        output_size (int): the size of the output
        dropout (float): the dropout rate
        use_gpu (bool): whether to use the GPU
        """
        super().__init__()
        
        self.config = config
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.latent_size = latent_size
        self.output_size = output_size
        self.dropout = dropout
        self.use_gpu = use_gpu

        self.rnn = nn.GRU(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            dropout=self.dropout,
            batch_first=True
        )

        self.rnn2out = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_size
        )
    
    def forward(self, embeddings, state, lengths):
        batch_size = embeddings.size(0)
        
        # packed is of shape (sum(lengths), embed_size)
        # lengths is a list of lengths for each sequence in the batch
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=True)

        # hidden is of shape (batch_size, seq_len, hidden_size)
        # state is of shape (hidden_layers, batch_size, hidden_size)
        hidden, state = self.rnn(packed, state)

        # transform state to shape (hidden_layers, batch_size, hidden_size)
        state = state.view(self.hidden_layers, batch_size, self.hidden_size)

        # pack the hidden states, shape (batch_size, seq_len, hidden_size)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)

        # output is of shape (batch_size, seq_len, output_size)
        output = self.rnn2out(hidden)

        return output, state

class Latent2Hidden(nn.Module):
    """
    This class is the linear layer that transforms the latent space to the hidden layers in the decoder.
    This class inherits from the nn.Module class in PyTorch.
    """
    def __init__(self, latent_size, hidden_size, hidden_layers, dropout, use_gpu):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.use_gpu = use_gpu

        self.latent2hidden = nn.Linear(
            in_features=self.latent_size,
            out_features=self.hidden_size * self.hidden_layers
            )

    def forward(self, z):
        batch_size = z.size(0)

        # hidden is of shape (batch_size, hidden_size * hidden_layers)
        hidden = self.latent2hidden(z)

        # hidden transformed to shape (batch_size, hidden_layers, hidden_size)
        hidden = hidden.view(batch_size, self.hidden_layers, self.hidden_size)
        
        # hidden transformed to shape (hidden_layers, batch_size, hidden_size)
        hidden = hidden.transpose(0, 1).contiguous()
        if self.use_gpu:
            hidden = hidden.cuda()

        return hidden

class MLP(nn.Module):
    """
    This class is the multi-layer perceptron (MLP) that predicts the logP and SAS scores.
    This class inherits from the nn.Module class in PyTorch.
    """
    def __init__(self, config, latent_size, hidden_size_mlp, hidden_layers_mlp, dropout, output_size = 1):
        super().__init__()
        self.config = config
        self.latent_size = latent_size
        self.hidden_size_mlp = hidden_size_mlp
        self.output_size = output_size
        self.hidden_layers_mlp = hidden_layers_mlp
        self.dropout = dropout
        self.logp = None
        self.sas = None

        if self.config.get('pred_logp'):
            self.layers_logp = self.create_layers(latent_size, hidden_size_mlp, hidden_layers_mlp, output_size)
                
        if self.config.get('pred_sas'):
            self.layers_sas = self.create_layers(latent_size, hidden_size_mlp, hidden_layers_mlp, output_size)

    def create_layers(self, input_size, hidden_size, hidden_layers, output_size):
        layers = nn.ModuleList()
        # input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=self.dropout))
        
        # hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout))
        
        # output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        return layers
    
    def forward_mlp(self, z, layers):
        for layer in layers:
            z = layer(z)
        return z
    
    def forward(self, z):
        if self.config.get('pred_logp'):
            # logp is of shape (batch_size)
            self.logp = self.forward_mlp(z, self.layers_logp).squeeze(-1)
        if self.config.get('pred_sas'):
            # sas is of shape (batch_size)
            self.sas = self.forward_mlp(z, self.layers_sas).squeeze(-1)
        return self.logp, self.sas
    
class Loss(nn.Module):
    """
    This class is the loss function of the VAE model. It is responsible for the computation of the loss.
    This class inherits from the nn.Module class in PyTorch.
    """
    def __init__(self, config, vocab, pad):
        super().__init__()
        self.config = config
        self.pad = pad
        self.vocab = vocab
        self.logp_loss = None
        self.sas_loss = None

    def forward(self, output, target, mu, sigma, pred_logp, labels_logp, pred_sas, labels_sas, epoch, penalty_weights, beta):
        output = F.log_softmax(output, dim=1)

        """# apply penalty weights
        target_pen_weight_lst = []
        for target_i in target.cpu().detach().numpy():
            target_pen_weight_i = penalty_weights[self.vocab.translate(target_i)].values
            if len(target_pen_weight_i) < target.size(1):
                pad_len = target.size(1) - len(target_pen_weight_i)
                target_pen_weight_i = np.pad(target_pen_weight_i, (0, pad_len), 'constant')
            target_pen_weight_lst.append(target_pen_weight_i)
        target_pen_weight = torch.Tensor(target_pen_weight_lst).view(-1)
        """

        target = target.view(-1)
        output = output.view(-1, output.size(2))

        # create a mask filtering out all tokens that ARE NOT the padding token
        mask = (target > self.pad).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).item())

        # pick the values for the label and zero out the rest with the mask
        #output = output[range(output.size(0)), target] * target_pen_weight.cuda() * mask
        output = output[range(output.size(0)), target] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        CE_loss = -torch.sum(output) / nb_tokens

        # compute KL Divergence
        KL_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

        if self.config.get('pred_logp'):
            # compute logp loss
            self.logp_loss = F.mse_loss(pred_logp.type(torch.float64), labels_logp)

        if self.config.get('pred_sas'):
            # compute sas loss
            self.sas_loss = F.mse_loss(pred_sas.type(torch.float64), labels_sas)
        if KL_loss > 10000000:
            total_loss = CE_loss
            if self.config.get('pred_logp'):
                total_loss += self.logp_loss
            if self.config.get('pred_sas'):
                total_loss += self.sas_loss
        else:
            total_loss = CE_loss + beta[epoch]*KL_loss
            if self.config.get('pred_logp'):
                total_loss += self.logp_loss
            if self.config.get('pred_sas'):
                total_loss += self.sas_loss
        return total_loss, CE_loss, KL_loss, self.logp_loss, self.sas_loss