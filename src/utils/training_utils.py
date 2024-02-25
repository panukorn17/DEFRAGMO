import torch
import pandas as pd

def load_checkpoint(trainer, last=True)->int:
    """
    Function that loads the latest checkpoint or a specific checkpoint

    Parameters:
    trainer (Trainer): The trainer object
    last (bool): Whether to load the last checkpoint or the best loss checkpoint
    """
    file_name = 'last.pt' if last is True else 'best_loss.pt'
    path  = trainer.config.path('ckpt') / file_name

    if trainer.config.get('use_gpu') is False:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(path)

    print(f'Loading {file_name} at epoch {checkpoint["epoch"]+1}...')

    trainer.model.load_state_dict(checkpoint['model'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler'])
    trainer.criterion.load_state_dict(checkpoint['criterion'])
    trainer.best_loss = checkpoint['best_loss']
    trainer.losses = checkpoint['losses']
    return checkpoint['epoch']

def save_checkpoint(trainer, epoch: int, file_name: str)->None:
    """
    Function that saves the latest checkpoint

    Parameters:
    trainer (VAETrainer): The trainer object
    epoch (int): The current epoch
    file_name (str): The name of the file to save the checkpoint
    """
    path  = trainer.config.path('ckpt') / file_name
    torch.save({
        'epoch': epoch,
        'model': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'scheduler': trainer.scheduler.state_dict(),
        'criterion': trainer.criterion.state_dict(),
        'best_loss': trainer.best_loss,
        'losses': trainer.losses
    }, path)

def get_optimizer(config, model)->torch.optim.Optimizer:
    """
    Function that returns the optimizer

    Parameters:
    config (Config): The configuration object
    model (nn.Module): The model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('optim_lr'))
    return optimizer

def get_scheduler(config, optimizer)->torch.optim.lr_scheduler._LRScheduler:
    """
    Function that returns the learning rate scheduler

    Parameters:
    config (Config): The configuration object
    optimizer (torch.optim.Optimizer): The optimizer
    """
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get('sched_step_size'), gamma=config.get('sched_gamma'))
    return scheduler

def dump(config, losses, CE_loss, KL_loss, pred_logp_loss, pred_sas_loss, beta_list):
    df = pd.DataFrame(list(zip(losses, CE_loss, KL_loss, beta_list)),
                      columns=["Total loss", "CE loss", "KL loss", "beta"])
    if config.get('pred_logp'):
        df["logP loss"] = pred_logp_loss
    if config.get('pred_sas'):
        df["SAS loss"] = pred_sas_loss
    filename = config.path('results') / "loss.csv"
    df.to_csv(filename)