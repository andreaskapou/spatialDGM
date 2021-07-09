"""
To run this template just do:
python vae.py
tensorboard --logdir default
"""
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

class VAE(pl.LightningModule):
    def __init__(self, hparams, data_dim: tuple, **kwargs):
        """
        Args:
            hparams: Parameters defined in the 'configs' folder
            data_dim: Dimension of each datapoint as a tuple
        """
        super(VAE, self).__init__()

        # Save hyperparameters for reproducibility
        self.save_hyperparameters()
        self.model = 'vae'
        self.hparams.update(hparams)
        self.lr = hparams.lr
        self.kl_coef = hparams.kl_coef
        self.data_dim = data_dim

        # Set activation function between layers
        if hparams.activation == 'relu':
            self.activation = nn.LeakyReLU
        elif hparams.activation == 'tanh':
            self.activation = nn.Tanh
        else: 
            self.activation = nn.LeakyReLU

        # Define encoder-decoder
        self.p_net = Decoder(data_dim=data_dim, latent_dim=hparams.z_dim, likelihood=hparams.likelihood,
                hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers, activation=self.activation)
        self.q_net = Encoder(data_dim=data_dim, latent_dim=hparams.z_dim, 
                hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers, activation=self.activation)

    def configure_optimizers(self):
        """
        Configure our default optimizer setting. PL will take care of everything else.
        """
        # Use Adam optimizer
        optim = torch.optim.Adam(
            params=list(self.p_net.parameters()) + list(self.q_net.parameters()), lr=self.lr)
        # Decay the learning rate of each parameter group by gamma
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optim, step_size=self.hparams.step_size, gamma=0.1)
        return [optim], [scheduler]

    def reparameterize(self, mu, logstd):
        """
        :param mu: mean from the encoder's latent space z
        :param logstd: log standard deviation from the encoder's latent space z
        """

        # Check also: https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py#L122
        # If we had as input logvar = log(s^2) = 2*logstd
        # Then std = exp(0.5 * logvar) = exp(0.5 * 2 * logstd)
        std = torch.exp(logstd)     # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size, sample from Normal
        z = mu + (eps * std)        # sampling as if coming from the input space
        return z, std

    def forward(self, y):
        """
        The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        Used for inference only (separate from training step)
        :param y: Input data point
        """

        # Encoder: from input to latent space
        z_mu, z_logstd = self.q_net(y)
        # Reparameterization trick so the error is backpropagated through the network
        z, z_std = self.reparameterize(mu=z_mu, logstd=z_logstd)
        # Decoder: from latent space to reconstructed input
        y_hat = self.p_net(z)
        return y_hat, z, z_mu, z_logstd, z_std


    def training_step(self, batch, batch_idx):
        """
        The full training loop operations
        """
        loss, elbo, log_p_x_q_z, kl_div, _ = self.shared_step(batch)

        ##
        # TODO: Need to understand how to take advantage of loggin and tensorboard
        ##
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return {'loss': loss,
                'log': {'elbo': elbo,
                        'log_p_x_q_z': log_p_x_q_z,
                        'kl_div': kl_div}}

    def validation_step(self, batch, batch_idx):
        """
        The full validation loop operations
        """
        loss, elbo, log_p_x_q_z, kl_div, y_hat = self.shared_step(batch)

        self.log('val_kl_div', kl_div, on_step=False, on_epoch=True)
        self.log('val_exp_log', log_p_x_q_z, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return y_hat, loss

    def validation_epoch_end(self, outputs):
        choice = random.choice(outputs)
        output_sample = choice[0]
        output_sample = output_sample.reshape(-1, *self.data_dim)
        save_path = f'../output/{self.model}_{self.hparams.dataset}_{self.hparams.modify}_z{self.hparams.z_dim}_images'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_image(output_sample, f"{save_path}/epoch_{self.current_epoch+1}.png")


    def shared_step(self, batch):
        """
        Training or validation step is implemented here
        """
        y = batch[0] # Get the batch of (modified) images
        
        # Run forward pass
        y_hat, _, z_mu, z_logstd, z_std = self.forward(y)

        # KL(q||p) = -0.5 * sum(1 + 2*log(s) - m^2 - s^2 )
        # https://ai.stackexchange.com/questions/26366/why-is-0-5-torch-sum1-sigma-mu-pow2-sigma-exp-in-pytorch-equiva
        kl_div = -0.5*(1 + 2*z_logstd - z_mu**2 - z_std**2).sum(dim = 1)
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

        # ELBO = E_q_z[log p(x|z)] - KL*beta, where alpha scaling coefficient (beta-VAE)
        elbo = log_p_x_q_z - kl_div*self.kl_coef
        loss = -elbo
        ##
        # TODO remove y_hat
        ##
        return loss, elbo, log_p_x_q_z, kl_div, y_hat 

    def compute_log_p_x_q_z(self, y_hat, y, size):
        """
        Compute expected log likelihood
        """
        if self.hparams.likelihood == 'bernoulli': 
            ##
            # TODO: why cross entropy with logits? and why do we multiply by size
            ##
            #log_p_x_q_z = -F.binary_cross_entropy_with_logits(input=y_hat, target=y, reduction='sum') * size
            log_p_x_q_z = -F.binary_cross_entropy(input=y_hat, target=y, reduction='sum')
        elif self.hparams.likelihood == 'gaussian':
            # -0.5 * torch.sum((y_hat - y)**2, 1).mean()
            log_p_x_q_z = -F.mse_loss(input=y_hat, target=y, reduction='sum')  
        else:
            raise ValueError(f'{self.hparams.likelihood} is not yet supported')
        return log_p_x_q_z


    def interpolate(self, x1, x2):
        """
        Not implemented / tested
        """
        assert x1.shape == x2.shape, "Inputs must be of the same shape"
        if x1.dim() == 3:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)
        if self.training:
            raise Exception(
                "This function should not be called when model is still "
                "in training mode. Use model.eval() before calling the "
                "function")

        mu1, lv1 = self.q_net(x1)
        mu2, lv2 = self.q_net(x2)

        z1, _ = self.reparameterize(mu=mu1, logstd=lv1)
        z2, _ = self.reparameterize(mu=mu2, logstd=lv2)
        # Decoder: from latent space to reconstructed input
        

        weights = torch.arange(0.1, 0.9, 0.1)
        intermediate = [self.decode(z1)]
        for wt in weights:
            inter = (1.-wt)*z1 + wt*z2
            intermediate.append(self.p_net(inter))
        intermediate.append(self.p_net(z2))
        out = torch.stack(intermediate, dim=0).squeeze(1)
        return out, (mu1, lv1), (mu2, lv2)


class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.LeakyReLU):
        super(ResidLinear, self).__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x) + x)


class Encoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden_dim, num_layers=1, activation=nn.LeakyReLU, resid=False):
        """
        The standard MLP structure for the encoder.
        """
        super(Encoder, self).__init__()
        self.data_dim = data_dim
        self.input_dim = np.prod(data_dim) # Total number of input features
        self.latent_dim = latent_dim

        # Define encoder structure
        layers = [nn.Linear(self.input_dim, hidden_dim),
                  activation(),
                  ]
        for _ in range(1, num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())
        # Store params for both mu and sigma, assuming N(mu, sigma)
        layers.append(nn.Linear(hidden_dim, 2 * latent_dim))

        # Combine all layers 
        self.layers = nn.Sequential(*layers)

    def forward(self, y):
        # y is (batch_size, data_dim)
        y_flat = torch.flatten(input = y, start_dim = 1)

        # Obtain latent parameters
        z = self.layers(y_flat)
        # Size is 2 * latent_dim to store mu and sigma
        z_mu = z[:, :self.latent_dim]
        z_logstd = z[:, self.latent_dim:]
        return z_mu, z_logstd

class Decoder(nn.Module):
    def __init__(self, data_dim, latent_dim, likelihood, hidden_dim, num_layers=1, activation=nn.LeakyReLU, resid=False):
        """
        The standard MLP structure for decoder. Decodes each pixel location as a funciton of z.
        """
        super(Decoder, self).__init__()
        self.data_dim = data_dim
        self.output_dim = np.prod(data_dim) # Total number of output features
        self.latent_dim = latent_dim

        layers = [nn.Linear(latent_dim, hidden_dim), 
                  activation()]
        for _ in range(1, num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        # If we expect Bernoulli likelihood, pass data through Sigmoid
        if likelihood == 'bernoulli':
            layers.append(nn.Sigmoid())

        # Combine all layers 
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        # z is (batch_size, latent_dim) --> ignores coordinates x, decodes each pixel conditioned on z.
        y = self.layers(z)
        # Reshape the output appropriately
        y = y.view(z.size(0), *self.data_dim)
        return y

