
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

GAME_BOARD_SIZE = 16
class VAE_DQN(nn.Module):
    def __init__(self):
        super(VAE_DQN, self).__init__()

        self.input_dim = GAME_BOARD_SIZE # game board is 4 * 4
        
        self.latent_dim = 20
        self.noise_scale = 0
        self.batch_size = 50

        self.encoder_l1 = nn.Sequential(
            torch.nn.Linear(self.input_dim, 400),
            #nn.BatchNorm1d(100)
            nn.ReLU()
        )
        '''
        self.encoder_l2 = nn.Sequential(
            torch.nn.Linear(400, 200),
            nn.ReLU()
        )
        self.encoder_l3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.encoder_l4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        '''
        self.fc_mu = nn.Linear(400, self.latent_dim)
        self.fc_sigma = nn.Linear(400, self.latent_dim)
        #self.fc_up = nn.Linear(self.latent_dim, 400)
        
        self.decoder_l1 = nn.Sequential(
            torch.nn.Linear(self.latent_dim, 400),
            nn.ReLU()
        )
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l2 = nn.Sequential(
            torch.nn.Linear(400, self.input_dim))
        #self.m1 = nn.MaxPool2d(3, stride=2)
        '''
        self.decoder_l3 = nn.Sequential(
            nn.ConvTranspose2d(33, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l4 = nn.Sequential(
            nn.ConvTranspose2d(17, 3, kernel_size=3, stride=1, padding=1))
	    #nn.Sigmoid())
        '''
        
        
    def encode(self, x):
        #print(x.shape)
        h1 = self.encoder_l1(x)
        #print(h1.shape)
        #h2 = self.encoder_l2(h1)
        #return self.fc_mu(h2), self.fc_sigma(h2)
        return self.fc_mu(h1), self.fc_sigma(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.decoder_l1(z)
        #return torch.sigmoid(self.fc4(h3))
        return self.decoder_l2(h3)

    def forward(self, x):
        #print(x.shape)
        mu, logvar = self.encode(x.view(-1, GAME_BOARD_SIZE))
        z = self.reparameterize(mu, logvar)
        #z = self.fc_up(z)
        #print(z.shape)
        return self.decode(z), mu, logvar
    


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #print(recon_x[0],x.view(-1, GAME_BOARD_SIZE)[0])
    #BCE = nn.MSELoss(recon_x, x.view(-1, GAME_BOARD_SIZE))
    recon_x = torch.ceil(recon_x)
    criterion = nn.MSELoss()
    BCE = criterion(recon_x, x.view(-1, GAME_BOARD_SIZE))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD