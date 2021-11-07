"""
To run this template just do:
python spatialVAE.py
tensorboard --logdir default
"""
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from vae import VAE, Encoder, Decoder, ResidLinear


class SpatialVAE(VAE):
    # Constructor
    def __init__(self, hparams, data_dim: tuple, **kwargs):
        """
        Args:
            hparams: Parameters defined in the configs folder
            data_dim: Dimension of each datapoint as a tuple
        """
        super(SpatialVAE, self).__init__(hparams=hparams, data_dim=data_dim, **kwargs)

        # Save hyperparameters for reproducibility
        self.save_hyperparameters()
        self.model_name = 'SpatialVAE'
        self.hparams.update(hparams)
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
        self.theta_prior = hparams.theta_prior
        # Standard deviation of 'translation' latent variables
        self.dx_scale = hparams.dx_scale

        # What modification we will perform on the data
        self.rotate = self.translate = False
        if hparams.modify == 1:
            self.rotate = True
        elif hparams.modify == 2:
            self.translate = True
        elif hparams.modify == 3:
            self.rotate = True
            self.translate = True

        # This is the latent space for encoder
        self.latent_dim = hparams.z_dim + self.rotate + 2 * self.translate 

        # Create fixed coordinates array x
        x0, x1 = np.meshgrid(np.linspace(-1, 1, data_dim[1]), np.linspace(1, -1, data_dim[2]))
        x = np.stack([x0.ravel(), x1.ravel()], 1)
        x = torch.from_numpy(x).float()
        self.x = x.cuda() if torch.cuda.is_available() else x.cpu()

        # Define encoder-decoder
        if self.modify == 0:
            self.p_net = Decoder(data_dim=data_dim, latent_dim=hparams.z_dim, likelihood=hparams.likelihood,
                hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers, activation=self.activation)
        else:
            self.p_net = SpatialDecoder(data_dim=data_dim, latent_dim=hparams.z_dim, likelihood=hparams.likelihood,
                hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers, activation=self.activation)
        self.q_net = Encoder(data_dim=data_dim, latent_dim=self.latent_dim, 
                hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers, activation=self.activation)


    def forward(self, y, fixed_theta=None):
        """
        The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        Used for inference only (separate from training step)
        :param y: Input data point
        :param x: The coordinates of the data [0,0],[0,0.01],...[1,1]
        :param fixed_theta: enforcing rotation angle
        """

        # Expand x to match batch size
        batch_size = y.size(0)
        x = self.x
        x = x.expand(batch_size, x.size(0), x.size(1))

        # Encoder: from input to latent space
        z_mu, z_logstd, z_std = self.q_net(y = y)
        # Reparameterization trick so the error is backpropagated through the network
        # Draw samples from variational posterior to calculate E[p(x|z)] 
        z = self.reparameterize(mu=z_mu, std=z_std)

        theta = dx = torch.tensor([0])
        if self.modify > 0:  # rotationally- and/or translationaly-invariant mode
            # Split latent variable into parts for rotation
            # and/or translation and image content
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

            if fixed_theta is not None:
                theta = torch.tensor([fixed_theta]).float()

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

        return y_hat, z, z_mu, z_logstd, z_std, theta

    def shared_step(self, batch):
        """
        Training or validation step is implemented here
        """
        y = batch[0] # Get the batch of (modified) images
        #x = self.x   # Extract fixed coordinates

        # Run forward pass
        y_hat, _, z_mu, z_logstd, z_std, _ = self.forward(y = y)

        kl_div = 0
        # z[0] is the latent variable that corresponds to the rotation
        if self.rotate:
            theta_mu, theta_std, theta_logstd = z_mu[:, 0], z_std[:, 0], z_logstd[:, 0]
            z_mu, z_std, z_logstd = z_mu[:, 1:], z_std[:, 1:], z_logstd[:, 1:]

            ##
            # TODO: Should there be a summation term as below for z?
            ##
            # Calculate the KL divergence term for rotation term
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            # KL(q || p) = -0.5 * ( 1 + 2*log(s) - 2*log(prior_std) - (m^2 + s^2) / (2*prior_std^2) )
            #kl_div = -0.5*(1 + 2*theta_logstd - 2*np.log(self.theta_prior) - 
            #    (theta_mu**2 + theta_std**2) / (2*self.theta_prior**2))
            kl_div = -0.5*(1 + 2*theta_logstd - 2*np.log(self.theta_prior) - 
                (theta_mu**2 + theta_std**2) / (self.theta_prior**2))


        ###### Unit normal prior over z (and translation) ####
        # KL(q||p) = -0.5 * sum(1 + 2*log(s) - m^2 - s^2 )
        # https://ai.stackexchange.com/questions/26366/why-is-0-5-torch-sum1-sigma-mu-pow2-sigma-exp-in-pytorch-equiva
        z_kl_div = -0.5 * (1 + 2*z_logstd - z_mu**2 - z_std**2).sum(dim = 1)

        # KL is the sum of terms for unstructured latent + structured latent (rotation)
        kl_div = kl_div + z_kl_div
        # Compute the mean KL across the batch
        kl_div = kl_div.mean(dim = 0)

        ##
        # TODO: Why do they multiply by size?
        ##
        # Compute expected log likelihood term (negative reconstruction loss)
        log_p_x_q_z = self.compute_log_p_x_q_z(
            y_hat=torch.flatten(y_hat, start_dim = 1), # (batch_size, data_dim)
            y=torch.flatten(y, start_dim = 1),         # (batch_size, data_dim)
            size=np.prod(self.data_dim))

        # ELBO = E_q_z[log p(x|z)] - KL*beta, where beta scaling coefficient (beta-VAE)
        elbo = log_p_x_q_z - kl_div*self.kl_coef
        loss = -elbo

        return loss, elbo, log_p_x_q_z, kl_div, y_hat # TODO remove y_hat

    def compute_log_p_x_q_z(self, y_hat, y, size):
        """
        Compute expected log likelihood. 
        Same as VAE class, adding here in case we want to change likelihood only for spatialVAE.
        """
        if self.hparams.likelihood == 'bernoulli': 
            ##
            # TODO: why cross entropy with logits? and why do we multiply by size
            ##
            #log_p_x_q_z = -F.binary_cross_entropy_with_logits(input=y_hat, target=y, reduction='sum') * size
            log_p_x_q_z = -F.binary_cross_entropy(input=y_hat, target=y, reduction='sum')# * size
        elif self.hparams.likelihood == 'gaussian':
            # -0.5 * torch.sum((y_hat - y)**2, 1).mean()
            log_p_x_q_z = -F.mse_loss(input=y_hat, target=y, reduction='sum')  
        else:
            raise ValueError(f'{self.hparams.likelihood} is not yet supported')
        return log_p_x_q_z

class SpatialDecoder(nn.Module):
    def __init__(self, data_dim, latent_dim, likelihood, hidden_dim, num_layers=1, activation=nn.LeakyReLU, resid=False):
        """
        The standard MLP structure for decoder. Decodes each pixel location as a function of z.
        """
        super(SpatialDecoder, self).__init__()
        self.data_dim = data_dim
        self.output_dim = 1          # Total number of output features, why 1??
        self.latent_dim = latent_dim

        # Linear layer for decoding spatial coordinates x
        in_dim = 2 # 2D x-y plane
        self.coord_linear = nn.Linear(in_dim, hidden_dim)

        # Why do we put bias=False for this layer? Is it because we added in coord_linear?
        if latent_dim > 0:
            self.latent_linear = nn.Linear(latent_dim, hidden_dim, bias=False)

        layers = [activation()]
        for _ in range(1, num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        # If we expect Bernoulli likelihood, pass data through softmax. 
        # Does the squashing make sense for the spatialVAE model?
        if likelihood == 'bernoulli':
           layers.append(nn.Sigmoid())

        # Combine all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x, z):
        # x is (batch_size, num_coordinates (size of dataset), 2 (x-y location))
        # z is (batch_size, latent_dim)

        if len(x.size()) < 3:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        n = x.size(1)

        ##
        # TODO: Why do they do this? This is a repetition across each batch
        ##
        # Flatten x so dim is now (batch_size * num_coordinates, 2)
        x = x.view(batch_size*n, -1)

        # Pass x coordinates through linear layer to obtain hidden space
        h_x = self.coord_linear(x)
        # Transform dim to (batch_size, n, hidden_dim)
        h_x = h_x.view(batch_size, n, -1)

        # We have one latent z, which will be applied to each coordinate
        h_z = 0
        if hasattr(self, 'latent_linear'):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            h_z = self.latent_linear(z)
            # Make it the same dim as h_x
            h_z = h_z.unsqueeze(1)

        # For each coordinate: add unstructed and structured components
        # Why do it this way to learn a joint hidden space?
        h = h_x + h_z  # (batch_size, num_coords, hidden_dim)
        
        # Again transform dimensions
        h = h.view(batch_size * n, -1)
        
        y = self.layers(h) # (batch_size*num_coords, output_dim), where output_dim = 1

        y = y.view(batch_size, n, -1) # (batch_size, num_coords, 1)

        # Num coordinates = data dimensions? that is why we do this here?
        # Reshape the output appropriately
        y = y.view(y.size(0), *self.data_dim)

        return y

