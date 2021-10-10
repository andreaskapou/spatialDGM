import sys
sys.path.append('../models')

import hydra
import logging
import os
import os.path as osp
import yaml

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

    # Store config file locally
    filename = open(osp.join(os.getcwd(), 'hparams.yaml'), "w")
    yaml.dump(OmegaConf.to_yaml(cfg), filename)
    filename.close()

    # To ensure reproducibility
    pl.seed_everything(123)

    # Load dataset, default we assume it is rotated
    dm = MnistDataModule(data_dir = osp.join('..', '..', 'data'), batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, modify=1)
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
    save_file = osp.join(os.getcwd(), 'VAE_{}_z{}_enc.pth'.format(cfg.dataset, cfg.z_dim))
    torch.save(vae_model.q_net.state_dict(), save_file)
    logger.info('Saving model: {}'.format(save_file))
    save_file = osp.join(os.getcwd(), 'VAE_{}_z{}_dec.pth'.format(cfg.dataset, cfg.z_dim))
    torch.save(vae_model.p_net.state_dict(), save_file)
    logger.info('Saving model: {}'.format(save_file))

if __name__ == "__main__":
    train_vae()
