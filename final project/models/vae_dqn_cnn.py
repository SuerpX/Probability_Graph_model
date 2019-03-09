
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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.input_dim = 17 # game board is 4 * 4
        
        self.latent_dim = LATENT_DIM
        self.noise_scale = 0
        self.batch_size = 64

        self.encoder_l1  =  nn.Sequential(
            nn.Conv2d(self.input_dim, 40, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.encoder_l2  =  nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.encoder_l3 = nn.Sequential(
            torch.nn.Linear(40 * GAME_BOARD_SIZE, 200),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(200, self.latent_dim)
        self.fc_sigma = nn.Linear(200, self.latent_dim)
        
        self.decoder_l1 = nn.Sequential(
            torch.nn.Linear(self.latent_dim, 200),
            nn.Tanh()
        )
        self.decoder_l2 = nn.Sequential(
            torch.nn.Linear(200, 40 * GAME_BOARD_SIZE),
            nn.ReLU())
        
        self.decoder_l3 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            )

        self.decoder_l4 = nn.Sequential(
            nn.ConvTranspose2d(40, self.input_dim, kernel_size=3, stride=1, padding=1)
            )
        
        
    def encode(self, x):
        #print(x.shape)
        h1 = self.encoder_l1(x)
        #print(h1.shape)
        h2 = self.encoder_l2(h1)

        h3 = self.encoder_l3(h2.view(-1, 40 * GAME_BOARD_SIZE))
        return self.fc_mu(h3), self.fc_sigma(h3)
        #return self.fc_mu(h1), self.fc_sigma(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h4 = self.decoder_l1(z)
        #print(h3.shape)
        h5 = self.decoder_l2(h4)
        #print(h4.shape)
        h5 = h5.view(-1, 40, 4, 4)
        h6 = self.decoder_l3(h5)
        h7 = self.decoder_l4(h6)
        #print(h4.shape)
        return torch.sigmoid(h7)

    def forward(self, x):
        #print(x.shape)
        mu, logvar = self.encode(x)#.view(-1, GAME_BOARD_SIZE))
        z = self.reparameterize(mu, logvar)
        #z = self.fc_up(z)
        #print(z.shape)
        return self.decode(z), mu, logvar, z

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
            torch.nn.Linear(100, 4),
            nn.Sigmoid()
            )
        self.fc1.apply(weights_initialize)
        self.fc2.apply(weights_initialize)
    def forward(self, z):
        x = self.fc1(z)
        q_values = self.fc2(x)
        return q_values

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
