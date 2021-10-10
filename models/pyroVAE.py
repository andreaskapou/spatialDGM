"""
To run this template just do:
python pyroVAE.py
tensorboard --logdir default
"""
import random
from typing import Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

from vae import Encoder, Decoder, ResidLinear
from spatialVAE import SpatialDecoder

from scipy.stats import norm
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist


def set_deterministic_mode(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PyroVAE(nn.Module):
    """
    Variational autoencoder with rotational and/or transaltional invariance in Pyro
    """
    def __init__(self, hparams, data_dim: tuple, **kwargs):
        """
        Args:
            hparams: Parameters defined in the configs folder
            data_dim: Dimension of each datapoint as a tuple
        """
        super(PyroVAE, self).__init__()

        pyro.clear_param_store()
        set_deterministic_mode(hparams.seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Save hyperparameters for reproducibility
        # self.save_hyperparameters()
        self.model_name = 'PyroVAE'
        self.hparams = hparams
        #self.hparams.update(hparams)
        self.lr = hparams.lr
        self.kl_coef = hparams.kl_coef
        self.data_dim = data_dim
        self.modify = hparams.modify

        # Set activation function between layers
        if hparams.activation == 'relu':
            self.activation = nn.LeakyReLU
        elif hparams.activation == 'tanh':
            self.activation = nn.Tanh
        else: 
            self.activation = nn.LeakyReLU

        ##
        # SpatialVAE specific params
        ##
        # Standard deviation on rotation prior theta
        self.theta_prior = torch.tensor(hparams.theta_prior).to(self.device)
        # Standard deviation of 'translation' latent variables
        self.dx_scale = torch.tensor(hparams.dx_scale).to(self.device)

        # What modification we will perform on the data
        self.rotate = self.translate = False
        if self.modify == 1:
            self.rotate = True
        elif self.modify == 2:
            self.translate = True
        elif self.modify == 3:
            self.rotate = True
            self.translate = True

        # This is the latent space for encoder
        self.latent_dim = hparams.z_dim + self.rotate + 2 * self.translate 

        # Create fixed coordinates array x
        x0, x1 = np.meshgrid(np.linspace(-1, 1, data_dim[1]), np.linspace(1, -1, data_dim[2]))
        x = np.stack([x0.ravel(), x1.ravel()], 1)
        self.x = torch.from_numpy(x).float().to(self.device)

        self.to(self.device)

        # Define encoder-decoder
        if self.modify == 0:
            self.p_net = Decoder(data_dim=data_dim, latent_dim=hparams.z_dim, likelihood=hparams.likelihood,
                hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers, activation=self.activation)
        else:
            self.p_net = SpatialDecoder(data_dim=data_dim, latent_dim=hparams.z_dim, likelihood=hparams.likelihood,
                hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers, activation=self.activation)
        self.q_net = Encoder(data_dim=data_dim, latent_dim=self.latent_dim, 
                hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers, activation=self.activation)


    def model(self, y: torch.Tensor) -> torch.Tensor:
        """
        Defines the model p(y|z)p(z)
        """
        # Register PyTorch Decoder module `p_net` with Pyro
        pyro.module("p_net", self.p_net)
        # KLD scale factor (see e.g. https://openreview.net/pdf?id=Sy2fzU9gl)
        beta = self.kl_coef
        # xy - spatial coordinates
        x = self.x

        data_dim = np.prod(self.data_dim)
        with pyro.plate("data", y.size(0)):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((y.size(0), self.latent_dim)))
            z_scale = x.new_ones(torch.Size((y.size(0), self.latent_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            with pyro.poutine.scale(scale=beta):
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            if self.modify > 0:  # rotationally- and/or translationaly-invariant mode
                # Split latent variable into parts for rotation
                # and/or translation and image content
                theta = dx = torch.tensor(0)
                if self.rotate & self.translate:
                    theta = z[:, 0]  # z[0] is the rotation theta
                    dx = z[:, 1:3]   # z[1:2] is the translation Dx
                    z = z[:, 3:]     # the remaining unstructured components
                elif self.rotate:
                    theta = z[:, 0]  # z[0] is the rotation theta
                    z = z[:, 1:]     # the remaining unstructured components
                elif self.translate:
                    dx = z[:, :2]    # z[0:1] is the translation Dx
                    z = z[:, 2:]     # the remaining unstructured components

                if self.rotate:
                    # Calculate the rotation matrix R
                    R = theta.data.new(batch_size, 2, 2).zero_()
                    R[:, 0, 0] = torch.cos(theta)
                    R[:, 0, 1] = torch.sin(theta)
                    R[:, 1, 0] = -torch.sin(theta)
                    R[:, 1, 1] = torch.cos(theta)
                    # Coordinate transformation by performing batch matrix-matrix multiplication
                    x = torch.bmm(x, R)  # rotate coordinates by theta

                if self.translate:
                    dx = dx * self.dx_scale # scale dx by standard deviation
                    dx = dx.unsqueeze(1)
                    # Translated coordinates by dx
                    x = x + dx 

            # Decoder: from latent space to reconstructed input
            if self.modify == 0:
                y_hat = self.p_net(z = z)
            else:
                # contiguous(): create copy where memory layout for elements is contiguous
                y_hat = self.p_net(x = x.contiguous(), z = z)

            # Observation model
            if self.hparams.likelihood == 'bernoulli': 
                pyro.sample(name="obs", 
                    fn=dist.Bernoulli(y_hat.view(-1, data_dim), validate_args=False).to_event(1),
                    obs=y.view(-1, data_dim))
            elif self.hparams.likelihood == 'gaussian':
                raise ValueError(f'{self.hparams.likelihood} is not yet supported')
            else: 
                raise ValueError(f'{self.hparams.likelihood} is not yet supported')

    def guide(self, y: torch.Tensor) -> torch.Tensor:
        """
        Defines the guide q(z|y)
        """
        # register PyTorch Encoder module `q_net` with Pyro
        pyro.module("q_net", self.q_net)
        # KLD scale factor (see e.g. https://openreview.net/pdf?id=Sy2fzU9gl)
        beta = self.kl_coef
        with pyro.plate("data", y.size(0)):
            # use the encoder to get the parameters used to define q(z|y)
            z_loc, z_logscale, z_scale = self.q_net(y)
            # sample the latent code z
            with pyro.poutine.scale(scale=beta):
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


    def _encode(self, y_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network
        in a batch-by-batch fashion
        """
        def inference() -> np.ndarray:
            with torch.no_grad():
                z, z_logscale, z_scale = self.q_net(y_i)
            encoded = torch.cat((z, z_scale), -1).cpu()
            return encoded

        y_new = y_new.to(self.device)
        num_batches = kwargs.get("num_batches", 10)
        batch_size = len(y_new) // num_batches
        z_encoded = []
        for i in range(num_batches):
            y_i = y_new[i*batch_size:(i+1)*batch_size]
            z_encoded_i = inference()
            z_encoded.append(z_encoded_i)
        y_i = y_new[(i+1)*batch_size:]
        if len(y_i) > 0:
            z_encoded_i = inference()
            z_encoded.append(z_encoded_i)
        return torch.cat(z_encoded)

    def encode(self, y_new: torch.Tensor, train_loader, **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network
        (this is baiscally a wrapper for self._encode)
        """
        if isinstance(y_new, torch.utils.data.DataLoader):
            y_new = train_loader.dataset.tensors[0]
        z = self._encode(y_new)
        z_loc = z[:, :self.latent_dim]
        z_scale = z[:, self.latent_dim:]
        return z_loc, z_scale


    def manifold2d(self, d: int, **kwargs: Union[str, int]) -> torch.Tensor:
        """
        Plots a learned latent manifold in the image space
        """
        grid_x = norm.ppf(torch.linspace(0.99, 0.01, d))
        grid_y = norm.ppf(torch.linspace(0.01, 0.99, d))
        y_hat_all = []
        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):
                z_sample = torch.tensor([xi, yi]).float().to(self.device).unsqueeze(0)
                #if self.num_classes > 0:
                #    z_sample = torch.cat([z_sample, cls], dim=-1)

                if self.modify == 0:
                    y_hat = self.p_net(z = z_sample).squeeze(1)
                else:
                    y_hat = self.p_net(x = self.x, z = z_sample).squeeze(1)

                y_hat_all.append(y_hat.detach().cpu())
        y_hat_all = torch.cat(y_hat_all)

        grid = make_grid(y_hat_all[:, None], nrow=d,
                         padding=kwargs.get("padding", 2),
                         pad_value=kwargs.get("pad_value", 0))
        plt.figure(figsize=(8, 8))
        plt.imshow(grid[0], cmap=kwargs.get("cmap", "gnuplot"),
                   origin=kwargs.get("origin", "upper"),
                   extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("$z_1$", fontsize=18)
        plt.ylabel("$z_2$", fontsize=18)
        plt.show()


class SVITrainer:
    """
    Stochastic variational inference (SVI) trainer for VAEs
    """
    def __init__(self, 
            model: Type[nn.Module],
            optimizer: Type[pyro.optim.PyroOptim] = None,
            loss: Type[pyro.infer.ELBO] = None,
            seed: int = 1
            ) -> None:
        """
        Initializes the trainer's parameters
        """
        pyro.clear_param_store()
        set_deterministic_mode(seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if optimizer is None:
            optimizer = pyro.optim.Adam({"lr": 1.0e-3})
        if loss is None:
            loss = pyro.infer.Trace_ELBO()

        self.svi = pyro.infer.SVI(model=model.model, guide=model.guide, optim=optimizer, loss=loss)
        self.loss_history = {"training_loss": [], "test_loss": []}
        self.current_epoch = 0

    def train(self, 
            train_loader: Type[torch.utils.data.DataLoader],
            **kwargs: float) -> float:
        """
        Trains a single epoch
        """
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch returned by the data loader
        for data in train_loader:
            y = data[0]
            loss = self.svi.step(y.to(self.device), **kwargs)
            # do ELBO gradient and accumulate loss
            epoch_loss += loss

        return epoch_loss / len(train_loader.dataset)

    def evaluate(self,
            test_loader: Type[torch.utils.data.DataLoader],
            **kwargs: float) -> float:
        """
        Evaluates current models state on a single epoch
        """
        # initialize loss accumulator
        test_loss = 0.
        # compute the loss over the entire test set
        with torch.no_grad():
            for data in test_loader:
                y = data[0]
                loss = self.svi.step(y.to(self.device), **kwargs)
                test_loss += loss

        return test_loss / len(test_loader.dataset)

    def step(self,
             train_loader: Type[torch.utils.data.DataLoader],
             test_loader: Optional[Type[torch.utils.data.DataLoader]] = None,
             **kwargs: float) -> None:
        """
        Single training and (optionally) evaluation step 
        """
        self.loss_history["training_loss"].append(self.train(train_loader,**kwargs))
        if test_loader is not None:
            self.loss_history["test_loss"].append(self.evaluate(test_loader,**kwargs))
        self.current_epoch += 1

    def print_statistics(self) -> None:
        """
        Prints training and test (if any) losses for current epoch
        """
        e = self.current_epoch
        if len(self.loss_history["test_loss"]) > 0:
            template = 'Epoch: {} Training loss: {:.4f}, Test loss: {:.4f}'
            print(template.format(e, self.loss_history["training_loss"][-1],
                                  self.loss_history["test_loss"][-1]))
        else:
            template = 'Epoch: {} Training loss: {:.4f}'
            print(template.format(e, self.loss_history["training_loss"][-1]))






