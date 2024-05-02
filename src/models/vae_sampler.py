class Sampler:
    """
    A class to sample from the VAE model.
    """
    def __init__(self, config, vocab, model):
        self.config = config
        self.vocab = vocab
        self.model = model