import sys
sys.path.append('../models')

from typing import Optional, Tuple, Type, Union
from functools import reduce
import operator
import yaml

import hydra
import logging
import os
import os.path as osp

import torch
import pyro

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from dm_mnist import MnistDataModule, MnistRotate
from pyroVAE import PyroVAE, SVITrainer

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def init_dataloader(*args: torch.Tensor, **kwargs: int
                        ) -> Type[torch.utils.data.DataLoader]:

        batch_size = kwargs.get("batch_size", 64)
        shuffle = kwargs.get("shuffle", True)
        num_workers = kwargs.get("num_workers", True)
        tensor_set = torch.utils.data.dataset.TensorDataset(*args)
        data_loader = torch.utils.data.DataLoader(
            dataset=tensor_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return data_loader


# Name of config file
config_name = 'pyroVAE_mnist'

@hydra.main(config_path=osp.join('..', 'configs'), config_name=config_name)
def train_vae(cfg):
    ###
    ## Configuration
    ###
    logger.info(f"Run config:\n{OmegaConf.to_yaml(cfg)}")
    out_dir = os.getcwd()
    logger.info('Working directory {}'.format(out_dir))

    # Store config file locally
    filename = open(osp.join(os.getcwd(), 'hparams.yaml'), "w")
    yaml.dump(OmegaConf.to_yaml(cfg), filename)
    filename.close()

    # To ensure reproducibility
    torch.manual_seed(cfg.seed)


    ###
    ##Load dataset, default we assume it is rotated
    ###
    dm = MnistDataModule(data_dir = osp.join('..', '..', 'data'), batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, modify=1)

    ####----------------------------------
    
    # TODO: Specific for current Pyro implementation. Need to make this more generic inside PyroVAE
    mnist_train_data = MnistRotate(root = osp.join('..', 'data'), train = True, modify=1,
                         transform=transforms.Compose([transforms.ToTensor()]))
    mnist_train_loader = torch.utils.data.DataLoader(
        mnist_train_data, batch_size=cfg.batch_size, shuffle=True)

    # Extract each component of the dataset in a different variable
    angles = []
    train_labels = []
    train_data = torch.zeros([0, 1, 28, 28])
    for datum, label, angle in mnist_train_loader:
        angles.append(angle.tolist())
        train_labels.append(label.tolist())
        train_data = torch.cat((train_data, datum), 0)

    angles = reduce(operator.concat, angles)
    train_labels = reduce(operator.concat, train_labels)
    train_data = torch.squeeze(train_data)

    # Create data loader
    train_loader = init_dataloader(
        train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    ###-------------------------------------


    # Size of each data point / image
    data_dim = (1, 28, 28)
    logger.info('Dataset size {}'.format(data_dim))

    ###
    ## Model
    ###
    vae_model = PyroVAE(hparams=cfg, data_dim=data_dim)

    ###
    ## Initialize SVI trainer
    ###
    trainer = SVITrainer(model=vae_model, optimizer=pyro.optim.Adam({"lr": cfg.lr}), 
        loss=pyro.infer.Trace_ELBO(), seed=cfg.seed)

    # Train for n epochs
    for e in range(cfg.num_epochs):
        trainer.step(train_loader)
        trainer.print_statistics()

    logger.info('Finished. Save to: {}'.format(os.getcwd()))

    ###
    ## Save models
    ###
    #save_file = osp.join(os.getcwd(), 'vae_{}_encoder.pth'.format(cfg.dataset))
    save_file = osp.join(os.getcwd(), 'PyroVAE_{}_m{}_z{}_enc.pth'.format(cfg.dataset, cfg.modify, cfg.z_dim))
    torch.save(vae_model.q_net.state_dict(), save_file)
    logger.info('Saving model: {}'.format(save_file))
    save_file = osp.join(os.getcwd(), 'PyroVAE_{}_m{}_z{}_dec.pth'.format(cfg.dataset, cfg.modify, cfg.z_dim))
    torch.save(vae_model.p_net.state_dict(), save_file)
    logger.info('Saving model: {}'.format(save_file))

if __name__ == "__main__":
    train_vae()
