from agents.agents import agent
import agents.agents
from random import randint, uniform
#from vae_dqn import loss_function
from memory import ReplayBuffer
from copy import deepcopy
import torch
from torch import nn, optim
import time
import numpy as np
from utility import normalization, oneHotMap, reverseOneHotMap, loss_function
import tensorflow as tf
device = torch.device("cuda")
class testAgent(agent):
    def __init__(self, model):
        super().__init__()
        #print(common.prioritized_memory.memory)
        self.step = 0
        self.vq = model
        self.loss = 0
        self.totalCorrect = 0
        self.acc = 0
        self.total = 0
        self.z_vector = []
    def action(self):
        self.step += 1
        a = self.actions[randint(0, 3)]
        board = deepcopy(self.gb.board)
        #print(board)
        #board[board == 0] = 1
        #board = np.log2(board)
        #print(board)
        board = oneHotMap(board)
        
        self.test(board)
        return a
    
    def test(self, board):
        
        #print(states[0])
        #print(board.shape)
        data = torch.from_numpy(board).type(torch.FloatTensor).cuda().view(-1, 17, 4, 4)
        #data = data.transpose(1, 2).view(-1, 17, 4, 4)

        recon_batch, mu, logvar = self.vq(data)

        #print(self.reparameterize(mu, logvar).shape)
        #self.z_vector.append()
        loss = loss_function(recon_batch, data, mu, logvar)

        board = reverseOneHotMap(board)
        #print(recon_batch.shape)
        #recon_batch = recon_batch.view(17, 16).transpose(0, 1)
        #print()
        target_board = reverseOneHotMap(recon_batch.data.cpu().numpy())
        #print(target_board.shape)
        self.totalCorrect += np.sum(board == target_board)
        self.total += 16

        self.acc = self.totalCorrect / self.total

        train_loss = loss.item()
        self.loss += loss.item() / len(data)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)