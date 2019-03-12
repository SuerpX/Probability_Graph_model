
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

GAME_BOARD_SIZE = 16
LATENT_DIM = 20

def weights_initialize(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
        module.bias.data.fill_(0.01)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.input_dim = LATENT_DIM
        self.fc1  =  nn.Sequential(
            torch.nn.Linear(self.input_dim, 100),
            nn.ReLU()#,
            #nn.BatchNorm1d(100)
        )

        self.fc2  =  nn.Sequential(
            torch.nn.Linear(100, 4)#,
            #nn.Sigmoid()
            )
        self.fc1.apply(weights_initialize)
        self.fc2.apply(weights_initialize)
    def forward(self, z):
        x = self.fc1(z)
        q_values = self.fc2(x)
        return q_values
"""
class VAE_DQN_CNN(nn.Module):
    def __init__(self):
        super(VAE_DQN_CNN, self).__init__()
        self.vae = VAE()
        self.dqn = DQN()
        self.dqn_loss = nn.SmoothL1Loss()

    def forward(self, x):
        x_hat, mu, logvar, z = self.vae(x)
        q_values = self.dqn(torch.exp(0.5*logvar).add(mu))

        return q_values, x_hat, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, target_q_values, q_values):
    #print(recon_x.shape)
    #print(x.shape)
        #print(type(recon_x), type(x), type(target_q_values), type(q_values))
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        dqn_loss = self.dqn_loss(q_values,target_q_values)
        #print(dqn_loss.item())

        return BCE + KLD + dqn_loss
"""
class VAE_DQN_CNN(nn.Module):
    def __init__(self, encoder):
        super(VAE_DQN_CNN, self).__init__()
        self.encoder = encoder
        self.dqn = DQN()
        self.dqn_loss = nn.SmoothL1Loss(reduction = 'sum')

    def forward(self, x):
        mu, logvar= self.encoder(x)
        q_values = self.dqn(torch.exp(0.5*logvar).add(mu))

        return q_values, mu, logvar

    def loss_function(self, target_q_values, q_values):

        dqn_loss = self.dqn_loss(q_values,target_q_values)
        
        return dqn_loss