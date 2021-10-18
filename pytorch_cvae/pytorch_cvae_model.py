from matplotlib.pyplot import yscale
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self, kernel_size, init_channels, image_channels, latent_dim):
        super(ConvVAE, self).__init__()

        self.kernel_size = kernel_size # (4, 4) kernel
        self.init_channels = init_channels # initial number of filters
        self.image_channels = image_channels # MNIST images are grayscale
        self.latent_dim = latent_dim # latent dimension for sampling

        # encoder
        encoder_dict = OrderedDict([
            ("enc1", nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1)),
            ("relu1", nn.ReLU()),
            ("enc2", nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1)),
            ("relu2", nn.ReLU()),
            ("enc3", nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1)),
            ("relu3", nn.ReLU()),
            ("enc4", nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
            stride=2, padding=0)),
            ("relu4", nn.ReLU()),            
        ])
        self.encoder = nn.Sequential(encoder_dict)

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)

        # decoder
        decoder_dict = OrderedDict([
            ("dec1", nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size,
            stride=1, padding=0)),
            ("relu1", nn.ReLU()),
            ("dec2", nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1)),
            ("relu2", nn.ReLU()),
            ("dec3", nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1)),
            ("relu3", nn.ReLU()),
            ("dec4", nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1)),   
            ("sigmoid", nn.Sigmoid()),         
        ])
        self.decoder = nn.Sequential(decoder_dict) 

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x, compute_loss=False):
        x = x.unsqueeze(1)
        # encoding
        x = self.encoder(x)
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)

        hidden = self.fc1(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)
 
        # decoding
        y = self.decoder(z)
        reconstruction = y.squeeze()

        if compute_loss:
            criterion = nn.BCELoss(reduction='sum')
            bce_loss = criterion(reconstruction, x)
            self.loss = self.final_loss(bce_loss, mu, log_var)

        return reconstruction
    
    def final_loss(self, bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the 
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD