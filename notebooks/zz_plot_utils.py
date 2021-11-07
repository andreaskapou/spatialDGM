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
import pyro
import pyro.distributions as dist

from dm_mnist import MnistDataModule
import matplotlib.pyplot as plt
from vae import VAE
from spatialVAE import SpatialVAE
from pyroVAE import PyroVAE
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
    enc_path = osp.join(base_dir, 'VAE_{}_z{}_enc.pth'.format(cfg.dataset, cfg.z_dim))
    dec_path = osp.join(base_dir, 'VAE_{}_z{}_dec.pth'.format(cfg.dataset, cfg.z_dim))
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
    vae = SpatialVAE(hparams=cfg, data_dim=data_dim)
    # Load pretrained model
    enc_path = osp.join(base_dir, 'SpatialVAE_{}_m{}_z{}_enc.pth'.format(cfg.dataset, cfg.modify, cfg.z_dim))
    dec_path = osp.join(base_dir, 'SpatialVAE_{}_m{}_z{}_dec.pth'.format(cfg.dataset, cfg.modify, cfg.z_dim))
    vae.q_net.load_state_dict(torch.load(enc_path, map_location=lambda storage, loc: storage), strict=False)
    vae.p_net.load_state_dict(torch.load(dec_path, map_location=lambda storage, loc: storage), strict=False)
    return vae

# Function for loading stored Pyro DGM
def load_pyrovae(base_dir: str, data_dim = (1, 28, 28)):
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    config = osp.join(base_dir, 'hparams.yaml')
    cfg = OmegaConf.load(config)
    #cfg = cfg.hparams
    vae = PyroVAE(hparams=cfg, data_dim=data_dim)
    # Load pretrained model
    enc_path = osp.join(base_dir, 'pyroVAE_{}_m{}_z{}_enc.pth'.format(cfg.dataset, cfg.modify, cfg.z_dim))
    dec_path = osp.join(base_dir, 'pyroVAE_{}_m{}_z{}_dec.pth'.format(cfg.dataset, cfg.modify, cfg.z_dim))
    vae.q_net.load_state_dict(torch.load(enc_path, map_location=lambda storage, loc: storage), strict=False)
    vae.p_net.load_state_dict(torch.load(dec_path, map_location=lambda storage, loc: storage), strict=False)
    return vae

def plot_latent(model, data, ax=None, num_batches=100):
    # Add the rotation parameter
    z_dim = 3 if model.hparams.modify > 0 else 2
    idx = 0 if z_dim == 2 else 1
    
    z = np.zeros((num_batches+1, z_dim))
    target = []
    angles = []
    for i, (img, t, angle) in enumerate(data):
        with torch.no_grad():
            zz, _, _ = model.q_net(img)
        #if model.model_name in ["SpatialVAE", "VAE"]:
        #    zz = model.reparameterize(mu=z_loc, std=z_scale)
        #elif model.model_name == "PyroVAE":
        #    zz = dist.Normal(z_loc, z_scale).sample()
        z[i, :] = zz[0].detach().numpy()
        target.append(t)
        angles.append(angle)
        if i >= num_batches:
            break
            
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize =(22, 6))
    im1 = ax1.scatter(z[:,idx], z[:,idx+1], c=angles, s=1.8, cmap='gnuplot2')
    ax1.set_xlabel("$z_1$", fontsize=14)
    ax1.set_ylabel("$z_2$", fontsize=14)
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=.8)
    cbar1.set_label("Angles (rad)", fontsize=14)
    cbar1.ax.tick_params(labelsize=10)

    im2 = ax2.scatter(z[:,idx], z[:,idx+1], c=target, s=1.8, cmap='tab10')
    ax2.set_xlabel("$z_1$", fontsize=14)
    ax2.set_ylabel("$z_2$", fontsize=14)
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=.8)
    cbar2.set_label("Labels", fontsize=14)
    cbar2.ax.tick_params(labelsize=10);
    #ax.scatter(z[:, 0], z[:, 1], c=target, cmap='tab10')
    
    im3 = ax3.scatter(angles, z[:,0], c=angles, s=1.8, cmap='gnuplot2')
    ax3.set_xlabel("True angle", fontsize=14)
    ax3.set_ylabel("Latent dim $z_1$", fontsize=14)
    cbar3 = fig.colorbar(im3, ax=ax3, shrink=.8)
    cbar3.set_label("Angles (rad)", fontsize=14)
    cbar3.ax.tick_params(labelsize=10);

    plt.show()
    
    
def plot_reconstructed(model, ax, r0=(-10, 10), r1=(-5, 10), n=15):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to('cpu')
            if model.hparams.modify > 0:
                x_hat = model.p_net(x=model.x, z=z)
            else:
                x_hat = model.p_net(z=z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    ax.imshow(img, extent=[*r0, *r1], cmap='gray')

### TODODO: Will need to update code here
####
###
def interpolate(model, x1, x2, n=12):
    z1, _, _ = model.q_net(x1)
    z2, _, _ = model.q_net(x2)
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

    
def plot_predictions(dm_loader, viz_idxs, axs, vae=None, spvae=None, pyrovae=None):
    for i, idx in enumerate(viz_idxs):
        # Get input and visualize it
        img, _, rot = dm_loader.dataset[idx]
        ax = axs[0, i]
        ax.imshow(img.squeeze(), cmap='gray')
        angle_deg = rot.item() * 180 / np.pi
        img_orig = TF.rotate(img=img, angle=-angle_deg) 
        ax = axs[1, i]
        ax.imshow(img_orig.squeeze(), cmap='gray')
        
        subplot_idx = 2
        # Inference vae
        if vae is not None:
            img_vae = vae.forward(img)[0].detach().numpy()
            ax = axs[subplot_idx, i]
            subplot_idx = subplot_idx + 1
            ax.imshow(img_vae.squeeze(), cmap='gray')

        # Inference spatial VAE
        if spvae is not None:
            img_spvae = spvae.forward(y=img)[0].detach().numpy()
            ax = axs[subplot_idx, i]
            subplot_idx = subplot_idx + 1
            ax.imshow(img_spvae.squeeze(), cmap='gray')
        
        # Inference Pyro VAE
        if pyrovae is not None:
            with torch.no_grad():
                img_pyrovae = pyrovae.reconstruct(y=img)[0].detach().numpy()
            ax = axs[subplot_idx, i]
            subplot_idx = subplot_idx + 1
            ax.imshow(img_pyrovae.squeeze(), cmap='gray')

    [ax.set_xticks([]) for ax in axs.flatten()]
    [ax.set_yticks([]) for ax in axs.flatten()]

    axs[0, 0].set_ylabel('Input')
    axs[1, 0].set_ylabel('Original')
    i = 2
    if vae is not None:
        axs[i, 0].set_ylabel('VAE')
        i = i+1
    if spvae is not None:
        axs[i, 0].set_ylabel('SpatialVAE')
        i = i+1
    if pyrovae is not None:
        axs[i, 0].set_ylabel('PyroVAE')
        i = i+1

    plt.tight_layout()
    plt.show()
    
    
    
def plot_predictions_pyro(dm_loader, viz_idxs, axs, vae=None, pyrovae=None):
    for i, idx in enumerate(viz_idxs):
        # Get input and visualize it
        img, _, rot = dm_loader.dataset[idx]
        ax = axs[0, i]
        ax.imshow(img.squeeze(), cmap='gray')
        angle_deg = rot.item() * 180 / np.pi
        img_orig = TF.rotate(img=img, angle=-angle_deg) 
        ax = axs[1, i]
        ax.imshow(img_orig.squeeze(), cmap='gray')
        
        subplot_idx = 2
        # Inference vae
        if vae is not None:
            with torch.no_grad():
                img_vae = vae.reconstruct(y=img)[0].detach().numpy()
            ax = axs[subplot_idx, i]
            subplot_idx = subplot_idx + 1
            ax.imshow(img_vae.squeeze(), cmap='gray')
        
        # Inference Pyro VAE
        if pyrovae is not None:
            with torch.no_grad():
                img_pyrovae = pyrovae.reconstruct(y=img)[0].detach().numpy()
            ax = axs[subplot_idx, i]
            subplot_idx = subplot_idx + 1
            ax.imshow(img_pyrovae.squeeze(), cmap='gray')
            
            with torch.no_grad():
                img_pyrovae = pyrovae.reconstruct_fully(y=img)[0].detach().numpy()
            ax = axs[subplot_idx, i]
            subplot_idx = subplot_idx + 1
            ax.imshow(img_pyrovae.squeeze(), cmap='gray')
            
            

    [ax.set_xticks([]) for ax in axs.flatten()]
    [ax.set_yticks([]) for ax in axs.flatten()]

    axs[0, 0].set_ylabel('Input')
    axs[1, 0].set_ylabel('Original')
    i = 2
    if vae is not None:
        axs[i, 0].set_ylabel('VAE')
        i = i+1
    if pyrovae is not None:
        axs[i, 0].set_ylabel('PyroVAE')
        i = i+1
        axs[i, 0].set_ylabel('PyroVAE full')

    plt.tight_layout()
    plt.show()