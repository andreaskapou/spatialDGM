import sys
sys.path.append('../models')

import hydra
import logging
import os
import os.path as osp
import pytorch_lightning as pl
import torch

from dm_mnist import MnistDataModule
from vae import VAE
from omegaconf import OmegaConf
logger = logging.getLogger(__name__)

# Name of config file
config_name = 'vae_mnist'

@hydra.main(config_path=osp.join('..', 'configs'), config_name=config_name)
def train_vae(cfg):
    logger.info(f"Run config:\n{OmegaConf.to_yaml(cfg)}")
    out_dir = os.getcwd()
    logger.info('Working directory {}'.format(out_dir))

    # To ensure reproducibility
    pl.seed_everything(123)

    # Dataset
    dm = MnistDataModule(data_dir = osp.join('..', '..', 'data'))
    logger.info('Dataset size {}'.format(dm.size()))

    # Model
    vae_model = VAE(cfg, data_dim=dm.size())

    # Train
    trainer = pl.Trainer(checkpoint_callback=False,
        max_epochs=cfg.num_epochs, fast_dev_run=cfg.fast_dev_run,
        gpus=[0] if torch.cuda.is_available() else 0)
    trainer.fit(model=vae_model, datamodule=dm)
    logger.info('Finished. Save to: {}'.format(os.getcwd()))

    # Save models
    #save_file = osp.join(os.getcwd(), 'vae_{}_encoder.pth'.format(cfg.dataset))
    save_file = osp.join(os.getcwd(), 'vae_{}_{}_z{}_encoder.pth'.format(cfg.dataset, cfg.modify, cfg.z_dim))
    torch.save(vae_model.q_net.state_dict(), save_file)
    logger.info('Saving model: {}'.format(save_file))
    save_file = osp.join(os.getcwd(), 'vae_{}_{}_z{}_decoder.pth'.format(cfg.dataset, cfg.modify, cfg.z_dim))
    torch.save(vae_model.p_net.state_dict(), save_file)
    logger.info('Saving model: {}'.format(save_file))

if __name__ == "__main__":
    train_vae()
