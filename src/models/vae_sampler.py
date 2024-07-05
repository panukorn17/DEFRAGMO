import torch 
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import itertools
import pickle
import os

from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles
from utils.mol_utils import mol_to_smiles, mols_from_smiles
from utils.config import RUNS_DIR
from data.fragmentation import replace_last, get_size
from data.fragment_embeddings import Vocabulary
from data.dataset import MoleculeFragmentsDataset


def remove_consecutive(fragments: list[str])->list[str]:
    """
    This function removes consecutive duplicate fragments from the list of fragments.

    Parameters:
    fragments (list): The list of fragments.

    Returns:
    list: The list of fragments with consecutive duplicates removed.

    Example: ['*C(C)(C)C', '*C(=O)Cc1coc2ccc(*)cc12', '*C(=O)Cc1coc2ccc(*)cc12', '*N*', '*N*', 'c1(*)ccccc1F', , '<EOS>']
    Output: ['*C(C)(C)C', '*C(=O)Cc1coc2ccc(*)cc12', '*N*', 'c1(*)ccccc1F', '<EOS>']
    """
    return [i for i, _ in itertools.groupby(fragments)]

def reconstruct(frags: list[str])->tuple[Chem.Mol, list[str]]:
    """
    This function reconstructs the molecule from the list of fragments.

    Parameters:
    frags (list): The list of fragments.

    Returns:
    tuple: The reconstructed molecule and the list of fragments.

    Example: ['*C(C)(C)C', '*C(=O)Cc1coc2ccc(*)cc12', '*N*', 'c1(*)ccccc1F']
    Output: (<rdkit.Chem.rdchem.Mol object at 0x000001A0C1E0EF80>, ['*C(C)(C)C', '*C(=O)Cc1coc2ccc(*)cc12', '*N*', 'c1(*)ccccc1F'])
    Note that the string representation of the molecule object is CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1
    """
    try:
        frags_test = frags.copy()
        frag_2_re = frags_test[-1]
        for i in range(len(frags_test)-1):
            frag_1_re = frags_test[-1*i-2]
            recomb = replace_last(frag_2_re, "*", frag_1_re.replace("*", "",1))
            recomb_canon = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(recomb)),rootedAtAtom = 1)
            frag_2_re = recomb_canon
        #print("Reconstruction successful")
        return MolFromSmiles(Chem.CanonSmiles(recomb)), frags
    except:
        #print("Reconstruction failed")
        return None, None

def generate_molecules(samples:np.ndarray, vocab:Vocabulary, check_eos:bool=False):
    """
    This function generates the molecules from the samples from the latent space.
    
    Parameters:
    samples (np.ndarray): The vocabulary integers sampled from the latent space.
    vocab (Vocabulary): The vocabulary object.
    check_eos (bool): Whether to check the end of sequence token.

    Returns:
    Depending on the check_eos parameter, it returns a list of tuples or a list of booleans.
    """
    result = []
    num_samples = samples.shape[0]
    if check_eos:
        running_check = [False] * num_samples

    for idx in range(num_samples):
        frag_smiles = vocab.translate(samples[idx, :])
        frag_smiles = remove_consecutive(frag_smiles)
        #print(frag_smiles)

        if len(frag_smiles) <= 1:
            if check_eos:
                running_check[idx] = True
                if idx == len(running_check)-1:
                    return running_check
                else:
                    continue
            continue

        try:
            frag_mols = mols_from_smiles(frag_smiles)
            mol, frags = reconstruct(frag_smiles)
            #print(mol)
            if mol is not None:
                smiles = mol_to_smiles(mol)
                if check_eos:
                    # count the number of "*" in smiles
                    count = sum(1 for c in smiles if c == '*')
                    # check if the number of atoms in the molecule is equal to the number of atoms in frag_smiles
                    count_atom_mol = get_size(mol)
                    count_atom_frags = sum([get_size(frag_mol) for frag_mol in frag_mols])
                    if (count > 0) & (idx < len(running_check)):
                        running_check[idx] = True
                        if idx == len(running_check)-1:
                            return running_check
                        else:
                            continue
                    elif (count == 0) & (idx < len(running_check)) & (count_atom_mol == count_atom_frags):
                        running_check[idx] = False
                        if idx == len(running_check)-1:
                            return running_check
                        else:
                            continue
                num_frags = len(frags)
                frags = " ".join(frags)
                result.append((smiles, frags, num_frags))
            else:
                if check_eos:
                    running_check[idx] = True
                    if idx == len(running_check)-1:
                        return running_check
                    else:
                        continue
                continue
        except Exception:
            if check_eos:
                running_check[idx] = True
                if idx == len(running_check)-1:
                    return running_check
                else:
                    continue
            continue
    if check_eos:
        return running_check
    else:
        return result

class Sampler:
    """
    A class to sample from the VAE model.
    """
    def __init__(self, config, vocab, model, var_const):
        self.config = config
        self.vocab = vocab
        self.model = model
        self.var_const = var_const
    
    def get_train_mean_std(self, dataset:MoleculeFragmentsDataset, num_training_points:int=None)->tuple[torch.Tensor, torch.Tensor]:
        """
        This method encodes the training data points and returns the mean and std of the latent space.

        Parameters:
        dataset (MoleculeFragmentsDataset): The dataset object.
        loader (torch.utils.data.DataLoader): The data loader.
        num_training_points (int): The number of training points to encode. If left empty, it encodes all the training points.

        Returns:
        torch.Tensor [latent_size]: The mean of the latent space.
        """
        print("Getting the mean and std of training set...")
        config_dir = RUNS_DIR / self.config.get('run_name') / 'config'
        # check if the mean and std files exist
        if os.path.exists(config_dir / 'mean.pkl') and os.path.exists(config_dir / 'std.pkl'):
            with open(config_dir / 'mean.pkl', 'rb') as f:
                mean = pickle.load(f)
            with open(config_dir / 'std.pkl', 'rb') as f:
                std = pickle.load(f)
            return mean, std
        self.model = self.model.cpu()
        loader = dataset.get_loader()
        self.model.eval()
        for idx, (src, tgt, lengths, data_index) in enumerate(loader):
            src, tgt = Variable(src), Variable(tgt)
            if self.config.get('use_gpu'):
                src, tgt = src.cuda(), tgt.cuda()

            if self.config.get('pred_logp') or self.config.get('pred_sas'):
                output, mu, sigma, z, logp, sas = self.model(src, lengths)
            else:
                output, mu, sigma, z = self.model(src, lengths)
            if idx == 0:
                z_stack = z.detach().cpu()
                mu_stack = mu.detach().cpu()
            else:
                z_stack = torch.vstack((z_stack, z.detach().cpu()))
                mu_stack = torch.vstack((mu_stack, mu.detach().cpu()))
            if idx%500 == 0:
                print(f"Encoded {len(z_stack)} points")
                torch.cuda.empty_cache()
            if num_training_points is not None:
                if len(z_stack) >= num_training_points:
                    z_stack = z_stack[:num_training_points]
                    break
        mean = torch.mean(z_stack, axis=0)
        std = torch.std(z_stack, axis=0)
        #save the mean as a pickle file
        with open(config_dir / 'mean.pkl', 'wb') as f:
            pickle.dump(mean, f)
        with open(config_dir / 'std.pkl', 'wb') as f:
            pickle.dump(std, f)
        return mean, std
    
    def sample(self, num_samples:int, train_mean:torch.Tensor, train_std:torch.Tensor)->list[str]:
        """
        This method sample from the latent space.

        Parameters:
        num_samples (int): The number of samples to generate.

        Returns:
        list: The list of generated molecules.
        """
        self.model = self.model.cpu()
        self.model.eval()
        vocab = self.vocab

        hidden_layers = self.model.hidden_layers
        hidden_size = self.model.hidden_size

        def row_filter(row):
            return (row == vocab.EOS).any()
        
        count = 0
        total_time = 0
        batch_size = 100
        c = self.var_const
        samples, sampled = [], 0

        max_length = self.config.get('max_len')
        temperature = self.config.get('temperature')
        sample_method = self.config.get('sampler_method')

        # expand the mean and std to the batch size
        mean = train_mean.expand(batch_size, -1)
        std = c * train_std.expand(batch_size, -1)

        with torch.no_grad():
            while len(samples) < num_samples:
                start = time.time()

                # sample vector from latent space
                z = torch.normal(mean, std).cpu()
                #z = self.model.encoder.sample_normal(batch_size)

                generated = self.sample_from_latent(z, hidden_layers, hidden_size, batch_size, max_length, temperature, sample_method, vocab)

                new_samples = generated.numpy()
                #print(new_samples)
                mask = np.apply_along_axis(row_filter, 1, new_samples)

                #check if any sequence is finished
                result = generate_molecules(new_samples[mask], vocab)
                samples.extend(result)

                end = time.time() - start
                total_time += end

                if len(samples) > sampled:
                    sampled = len(samples)
                    count = 0 
                else: 
                    count += 1

                if len(samples) % 100 < 10:
                    elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
                    print(f'Sampled {len(samples)} molecules. '
                          f'Time elapsed: {elapsed}')

                if count >= 1000000:
                    break

        return samples
    
    def sample_from_latent(self, z:torch.Tensor, hidden_layers:int, hidden_size:int, batch_size:int, max_length:int, temperature:float, sample_method:str, vocab:Vocabulary)->torch.Tensor:
        """
        This method samples from the latent space, calculates the probabilities for the next token for each sequence in the batch and ensures 
        that the generated sequences produce a valid molecule. The method returns the generated sequences.

        Parameters:
        z (torch.Tensor): The tensor of latent space vectors.
        hidden_layers (int): The number of hidden layers.
        hidden_size (int): The hidden size.
        batch_size (int): The batch size.
        max_length (int): The maximum length of the sequence.
        temperature (float): The temperature for sampling.
        sample_method (str): The sampling method either 'greedy', 'sample_first' or 'sample_all'.
        vocab (Vocabulary): The vocabulary object.

        Returns:
        torch.Tensor [batch_size, max_length]: The generated sequences.
        """
        # get the initial state
        state = self.model.latent2hidden(z)
        state = state.view(hidden_layers, batch_size, hidden_size)

        # all idx of batch
        sequence_idx = torch.arange(0, batch_size).long()

        # all idx of batch which are still generating
        running = torch.arange(0, batch_size).long()
        sequence_mask = torch.ones(batch_size, dtype=torch.bool)

        # idx of still generating sequences
        # with respect to current loop
        running_seqs = torch.arange(0, batch_size).long()
        lengths = [1] * batch_size

        generated = torch.Tensor(batch_size, max_length).long()
        generated.fill_(vocab.PAD)

        inputs = Variable(torch.Tensor(batch_size).long())
        inputs.fill_(vocab.SOS).long()

        step = 0

        while(step < max_length and len(running_seqs) > 0):
            prob_tracker = torch.zeros(len(running))
            inputs = inputs.unsqueeze(1)
            emb = self.model.embedder(inputs)
            scores, state = self.model.decoder(emb, state, lengths)
            scores = scores.squeeze(1)

            probs = F.softmax(scores / temperature, dim=1)
            
            if sample_method == 'greedy':
                # argmax
                inputs = torch.argmax(probs, 1).reshape(1, -1)
            elif sample_method == 'sample_first':
                # test sampling first token
                if step == 0:
                    inputs = torch.multinomial(probs, 1, replacement = True).reshape(1, -1)
                else:
                    inputs = torch.argmax(probs, 1).reshape(1, -1)
            elif sample_method == 'sample_all': 
                # sample unless the argmax sample is EOS
                inputs = torch.argmax(probs, 1).reshape(1, -1)
                inputs[inputs!=vocab.EOS] = torch.multinomial(probs[(inputs!=vocab.EOS)[0]], 1).reshape(1, -1)
            if (inputs == vocab.EOS).any():
                # global running sequences to check
                sequence_mask_check = torch.ones(batch_size, dtype=torch.bool)
                sequence_mask_check[running] = (inputs == vocab.EOS)
                running_check = running.masked_select(sequence_mask_check[running])
                # check reconstruction
                sequence_mask_check_recon = generate_molecules(generated[running_check].numpy(), vocab, check_eos=True)
                # update the global running sequences
                sequence_mask_check[running_check] = torch.tensor(sequence_mask_check_recon)
                # update local running sequences to resample
                running_mask_check = torch.arange(0, len(probs)).long()
                # resample if sequence_mask_check_recon is False
                running_resample = running_mask_check.masked_select(sequence_mask_check[running])
                # add 1 to the prob_tracker tensor for each index that needs to be resampled
                prob_tracker[running_resample] += 1
                # check the second highest prob for each index by running_resample
                # initialise empty tensor to store the n highest prob
                indices_tracker = torch.zeros(len(running_resample))
                for i in range(len(running_resample)):
                    # get the second highest prob
                    _, indices = torch.topk(probs[running_resample[i]], int(prob_tracker[running_resample[i]]) + 1, dim=0)
                    inputs[0, running_resample[i]] = indices[-1]
                    indices_tracker[i] = indices[-1]
                #_, indices = torch.topk(probs[running_resample], 2, dim=1)
                # replace the EOS token with the second highest prob
                #inputs[0, running_resample] = indices[:, 1]
                #inputs[0, running_resample] = indices_tracker
            # if this set of inputs is the same as any of the previous inputs
            if step>0:
                if self.config.get('sample_repeat') == 'unique_all':
                    for i in range(len(running)):
                        # update prob_tracker if the input is the same as any of the previous inputs
                        while (inputs[0, i] == generated[running[i]]).any():
                            prob_tracker[i] += 1
                            _, indices = torch.topk(probs[i], int(prob_tracker[i]) + 1, dim=0)
                            inputs[0, i] = indices[-1]
                elif self.config.get('sample_repeat') == 'unique_adjacent':
                    for i in range(len(running)):
                        if inputs[0, i] == generated[running[i], step-1]:
                            prob_tracker[i] += 1
                            _, indices = torch.topk(probs[i], int(prob_tracker[i]) + 1, dim=0)
                            inputs[0, i] = indices[-1]

            # save next input
            generated = self.update(generated, inputs, running, step)
            # update global running sequence
            sequence_mask[running] = (inputs != vocab.EOS)
            # check reconstruction when EOS is generated
            running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (inputs != vocab.EOS)
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            run_length = len(running_seqs)
            if run_length > 0:
                inputs = inputs.squeeze(0)
                inputs = inputs[running_seqs]
                state = state[:, running_seqs]
                running_seqs = torch.arange(0, run_length).long()

            lengths = [1] * run_length
            step += 1
        return generated

    def update(self, save_to:torch.Tensor, sample:torch.Tensor, running_seqs:torch.Tensor, step:int)->torch.Tensor:
        """
        This method updates the generated tensor with the new samples.

        Parameters:
        save_to (torch.Tensor): The tensor to save the samples to.
        sample (torch.Tensor): The new samples.
        running_seqs (torch.Tensor): The indices of the running sequences.
        step (int): The step in the sequence.

        Returns:
        torch.Tensor [batch_size, max_length]: The updated tensor.
        """
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at step position
        running_latest[:, step] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
