
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

GAME_BOARD_SIZE = 16
LATENT_DIM = 20

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.input_dim = 17 # game board is 4 * 4
        
        self.latent_dim = LATENT_DIM

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

    def encode(self, x):
        #print(x.shape)
        h1 = self.encoder_l1(x)
        #print(h1.shape)
        h2 = self.encoder_l2(h1)

        h3 = self.encoder_l3(h2.view(-1, 40 * GAME_BOARD_SIZE))
        return self.fc_mu(h3), self.fc_sigma(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        self.output_dim = 17 # game board is 4 * 4
        
        self.latent_dim = LATENT_DIM

        super(Decoder, self).__init__()
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
            nn.ConvTranspose2d(40, self.output_dim, kernel_size=3, stride=1, padding=1)
            )
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
        return self.decode(x)

class VAE_CNN(nn.Module):
    def __init__(self):
        super(VAE_CNN, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    def forward(self, x):
        #print(x.shape)
        mu, logvar = self.encoder(x)#.view(-1, GAME_BOARD_SIZE))
        z = self.reparameterize(mu, logvar)
        #z = self.fc_up(z)
        #print(z.shape)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        #print(recon_x.shape)
        #print(x.shape)
        #print(type(recon_x), type(x), type(target_q_values), type(q_values))
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

"""
GAME_BOARD_SIZE = int(16 / 4)
class VAE_DQN_CNN(nn.Module):
    def __init__(self):
        super(VAE_DQN_CNN, self).__init__()

        self.input_dim = 17 # game board is 4 * 4
        
        self.latent_dim = 20
        self.noise_scale = 0
        self.batch_size = 64

        self.encoder_l1  =  nn.Sequential(
            nn.Conv2d(self.input_dim, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            )

        self.encoder_l2  =  nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            )

        self.encoder_l3 = nn.Sequential(
            torch.nn.Linear(128 * GAME_BOARD_SIZE, 200),
            nn.Tanh(),
            nn.BatchNorm1d(200)
        )
        '''
        self.encoder_l2 = nn.Sequential(
            torch.nn.Linear(400, 200),
            nn.ReLU()
        )
        self.encoder_l3
        self.encoder_l4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        '''
        self.fc_mu = nn.Linear(200, self.latent_dim)
        self.fc_sigma = nn.Linear(200, self.latent_dim)
        #self.fc_up = nn.Linear(self.latent_dim, 400)
        
        self.decoder_l1 = nn.Sequential(
            torch.nn.Linear(self.latent_dim, 200),
            nn.Tanh(),
            nn.BatchNorm1d(200)
        )
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l2 = nn.Sequential(
            torch.nn.Linear(200, 128 * GAME_BOARD_SIZE),
            nn.ReLU(),
            nn.BatchNorm1d(128 * GAME_BOARD_SIZE)
            )
        #self.m1 = nn.MaxPool2d(3, stride=2)
        
        self.decoder_l3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            )

        self.decoder_l4 = nn.Sequential(
            nn.ConvTranspose2d(64, self.input_dim, kernel_size=2, stride=1)
            )
        '''
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l4 = nn.Sequential(
            nn.ConvTranspose2d(17, 3, kernel_size=3, stride=1, padding=1))
        #nn.Sigmoid())
        '''
        
        
    def encode(self, x):
        #print(x.shape)
        h1 = self.encoder_l1(x)
        #print(h1.shape)
        h2 = self.encoder_l2(h1)
        #print(h2.shape)
       # print(h2.view(-1, 40 * GAME_BOARD_SIZE).shape)

        h3 = self.encoder_l3(h2.view(-1, 128 * GAME_BOARD_SIZE))
        #print(h3.shape)
        return self.fc_mu(h3), self.fc_sigma(h3)
        #return self.fc_mu(h1), self.fc_sigma(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        #print(z.shape)
        h4 = self.decoder_l1(z)
        #print(h4.shape)
        h5 = self.decoder_l2(h4)
        #print(h5.shape)
        h5 = h5.view(-1, 128, 2, 2)
        #print(h5.shape)
        h6 = self.decoder_l3(h5)
        #print(h6.shape)
        h7 = self.decoder_l4(h6)
        #print(h7.shape)
        #print(h4.shape)
        #return torch.sigmoid(self.fc4(h3))
        #return torch.sigmoid(self.decoder_l2(h3))
        return torch.sigmoid(h7)

    def forward(self, x):
        #print(x.shape)
        mu, logvar = self.encode(x)#.view(-1, GAME_BOARD_SIZE))
        z = self.reparameterize(mu, logvar)
        #z = self.fc_up(z)
        #print(z.shape)
        return self.decode(z), mu, logvar

GAME_BOARD_SIZE = 16
class VAE_DQN_CNN(nn.Module):
    def __init__(self):
        super(VAE_DQN_CNN, self).__init__()

        self.input_dim = 17 # game board is 4 * 4
        
        self.latent_dim = 20
        self.noise_scale = 0
        self.batch_size = 64

        self.encoder_l1  =  nn.Sequential(
            nn.Conv2d(self.input_dim, 40, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU())

        self.encoder_l2  =  nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU())

        self.encoder_l3 = nn.Sequential(
            torch.nn.Linear(40 * GAME_BOARD_SIZE, 200),
            #nn.BatchNorm1d(100)
            nn.Tanh()
        )
        '''
        self.encoder_l2 = nn.Sequential(
            torch.nn.Linear(400, 200),
            nn.ReLU()
        )
        self.encoder_l3
        self.encoder_l4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        '''
        self.fc_mu = nn.Linear(200, self.latent_dim)
        self.fc_sigma = nn.Linear(200, self.latent_dim)
        #self.fc_up = nn.Linear(self.latent_dim, 400)
        
        self.decoder_l1 = nn.Sequential(
            torch.nn.Linear(self.latent_dim, 200),
            nn.Tanh()
        )
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l2 = nn.Sequential(
            torch.nn.Linear(200, 40 * GAME_BOARD_SIZE),
            nn.ReLU())
        #self.m1 = nn.MaxPool2d(3, stride=2)
        
        self.decoder_l3 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, kernel_size=3, stride=1, padding=1),
           #nn.BatchNorm2d(32),
            nn.ReLU()
            )

        self.decoder_l4 = nn.Sequential(
            nn.ConvTranspose2d(40, self.input_dim, kernel_size=3, stride=1, padding=1)
            )
        '''
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l4 = nn.Sequential(
            nn.ConvTranspose2d(17, 3, kernel_size=3, stride=1, padding=1))
	    #nn.Sigmoid())
        '''
        
        
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
        #return torch.sigmoid(self.fc4(h3))
        #return torch.sigmoid(self.decoder_l2(h3))
        return torch.sigmoid(h7)

    def forward(self, x):
        #print(x.shape)
        mu, logvar = self.encode(x)#.view(-1, GAME_BOARD_SIZE))
        z = self.reparameterize(mu, logvar)
        #z = self.fc_up(z)
        #print(z.shape)
        return self.decode(z), mu, logvar
"""
