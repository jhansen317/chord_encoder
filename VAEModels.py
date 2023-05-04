import torch
from torch import nn
import json
import torch.nn.functional as F

def kl_divergence(rho, rho_hat):
    return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

class Decoder(nn.Module):
    def __init__(self, **config):
        super(Decoder, self).__init__()
        hidden_activation_name = config.get('hidden_activation', 'tanh')
        if hidden_activation_name == 'tanh':
            hidden_activation = nn.Tanh
        elif hidden_activation_name == 'leaky_relu':
            hidden_activation = nn.LeakyReLU
        elif hidden_activation_name == 'relu':
            hidden_activation = nn.ReLU
        
        default_dims = [128]
        hidden_dims = sorted(config.get('hidden_dims', default_dims))

        modules = []
        self.decoder_input = nn.Sequential(
                    nn.Linear(config['latent_dim'], hidden_dims[0]),
                    nn.BatchNorm1d(hidden_dims[0]),
                    #nn.Dropout1d(p=0.2),
                    hidden_activation())
        self.decoder_output = nn.Sequential(
                    nn.Linear(hidden_dims[-1],  config['data_length']),
                    nn.Sigmoid())
        modules.append(self.decoder_input)
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    #nn.Dropout1d(p=0.2),
                    hidden_activation())
            )
        modules.append(self.decoder_output)
        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        z = self.decoder(z)
        return z

class VariationalEncoder(nn.Module):
    def __init__(self, **config):
        super(VariationalEncoder, self).__init__()
        hidden_activation_name = config.get('hidden_activation', 'tanh')
        if hidden_activation_name == 'tanh':
            hidden_activation = nn.Tanh
        elif hidden_activation_name == 'leaky_relu':
            hidden_activation = nn.LeakyReLU
        elif hidden_activation_name == 'relu':
            hidden_activation = nn.ReLU
        default_dims = [128]
        hidden_dims = sorted(config.get('hidden_dims', default_dims), reverse=True)
        modules = []
        self.encoder_input = nn.Sequential(
                    nn.Linear(config['data_length'], hidden_dims[0]),
                    nn.BatchNorm1d(hidden_dims[0]),
                    #nn.Dropout1d(p=0.2),
                    hidden_activation())
        modules.append(self.encoder_input)
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    #nn.Dropout1d(p=0.2),
                    hidden_activation())
            )
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], config['latent_dim'])
        self.fc_var = nn.Linear(hidden_dims[-1], config['latent_dim'])
        self.N = torch.distributions.Normal(0, 1)
        #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        #self.N.scale = self.N.scale.cuda()
        self.kl = 0


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        mu =  self.fc_mu(x)
        sigma = torch.exp(self.fc_var(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2)
        return z

class GaussianVAE(nn.Module):
    
    def __init__(self, **config):
        super(GaussianVAE, self).__init__()
        self.config = config
        self.encoder = VariationalEncoder(**self.config)
        self.decoder = Decoder(**self.config)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
