# Load libraries
import sys
sys.path.append('../models')
import os.path as osp
import yaml

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from dm_mnist import MnistDataModule
import matplotlib.pyplot as plt
from vae import VAE
from spatialVAE import spatialVAE
from omegaconf import OmegaConf

# Function for loading stored DGM
def load_vae(base_dir: str, data_dim = (1, 28, 28)):
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    config = osp.join(base_dir, 'lightning_logs/version_0/hparams.yaml')
    cfg = OmegaConf.load(config)
    cfg = cfg.hparams
    vae = VAE(hparams=cfg, data_dim=data_dim)
    # Load pretrained model
    enc_path = osp.join(base_dir, 'vae_{}_{}_z{}_encoder.pth'.format(cfg.dataset, cfg.modify, cfg.z_dim))
    dec_path = osp.join(base_dir, 'vae_{}_{}_z{}_decoder.pth'.format(cfg.dataset, cfg.modify, cfg.z_dim))
    vae.q_net.load_state_dict(torch.load(enc_path, map_location=lambda storage, loc: storage), strict=False)
    vae.p_net.load_state_dict(torch.load(dec_path, map_location=lambda storage, loc: storage), strict=False)
    return vae

# Function for loading stored DGM
def load_spvae(base_dir: str, data_dim = (1, 28, 28)):
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    config = osp.join(base_dir, 'lightning_logs/version_0/hparams.yaml')
    cfg = OmegaConf.load(config)
    cfg = cfg.hparams
    vae = spatialVAE(hparams=cfg, data_dim=data_dim)
    #vae = nn.DataParallel(vae)
    # Load pretrained model
    enc_path = osp.join(base_dir, 'spatialVAE_{}_{}_z{}_encoder.pth'.format(cfg.dataset, cfg.modify, cfg.z_dim))
    dec_path = osp.join(base_dir, 'spatialVAE_{}_{}_z{}_decoder.pth'.format(cfg.dataset, cfg.modify, cfg.z_dim))
    vae.q_net.load_state_dict(torch.load(enc_path, map_location=lambda storage, loc: storage), strict=False)
    vae.p_net.load_state_dict(torch.load(dec_path, map_location=lambda storage, loc: storage), strict=False)
    return vae

def plot_latent(model, data, ax=None, z_dim=2, num_batches=100):
    z = np.zeros((num_batches+1, z_dim))
    target = []
    for i, (img, t, angle) in enumerate(data):
        # Second parameter of forward is z: latent space
        if model.__class__.__name__ == "spatialVAE":
            z[i, :] = model(y=img, x=model.x)[1].detach().numpy()
        elif model.__class__.__name__ == "VAE":
            z[i, :] = model.forward(img)[1].detach().numpy()
        target.append(t)
        if i >= num_batches:
            break
    ax.scatter(z[:, 0], z[:, 1], c=target, cmap='tab10')
    
    
def plot_reconstructed(model, ax, r0=(-10, 10), r1=(-5, 10), n=15):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]])#.to('cpu')
            if model.__class__.__name__ == "spatialVAE":
                x_hat = model.p_net(x = model.x, z = z)
            elif model.__class__.__name__ == "VAE":
                x_hat = model.p_net(z = z)
            x_hat = x_hat.reshape(28, 28).detach().numpy() #.to('cpu')
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    ax.imshow(img, extent=[*r0, *r1], cmap='gray')
    
    
def interpolate(model, x1, x2, n=12):
    mu1, lv1 = model.q_net(x1)
    mu2, lv2 = model.q_net(x2)
    z1, _ = model.reparameterize(mu=mu1, logstd=lv1)
    z2, _ = model.reparameterize(mu=mu2, logstd=lv2)
    
    z = torch.stack([z1 + (z2 - z1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = model.p_net(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 28
    img = np.zeros((w, n*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

    
def plot_predictions(dm_loader, viz_idxs, axs, vae, spvae):
    for i, idx in enumerate(viz_idxs):
        # Get input and visualize it
        img, _, rot = dm_loader.dataset[idx]
        ax = axs[0, i]
        ax.imshow(img.squeeze(), cmap='gray')
        angle_deg = rot.item() * 180 / np.pi
        img_orig = TF.rotate(img=img, angle=-angle_deg) 
        ax = axs[1, i]
        ax.imshow(img_orig.squeeze(), cmap='gray')

        # Inference vae
        img_vae = vae.forward(img)[0].detach().numpy()
        ax = axs[2, i]
        ax.imshow(img_vae.squeeze(), cmap='gray')

        # Inference vae
        img_spvae = spvae.forward(y=img, x=spvae.x)[0].detach().numpy()
        ax = axs[3, i]
        ax.imshow(img_spvae.squeeze(), cmap='gray')

    [ax.set_xticks([]) for ax in axs.flatten()]
    [ax.set_yticks([]) for ax in axs.flatten()]

    axs[0, 0].set_ylabel('Input')
    axs[1, 0].set_ylabel('Original')
    axs[2, 0].set_ylabel('VAE')
    axs[3, 0].set_ylabel('spatialVAE')

    plt.tight_layout()
    plt.show()