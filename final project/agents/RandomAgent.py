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
class randomAgent(agent):
    def __init__(self, model, opt):
        super().__init__()
        #print(common.prioritized_memory.memory)
        self.memory = ReplayBuffer(500)
        self.previous_state = None
        self.previous_action = None
        self.step = 0
        self.vq = model
        self.loss = 0
        self.batch_size = 64
        self.opt = opt
    def action(self):
        self.step += 1
        a = self.actions[randint(0, 3)]
        board = deepcopy(self.gb.board)
        #print(board)
        #board[board == 0] = 1
        #board = np.log2(board)
        #print(board)
        board = oneHotMap(board)
        #board = board.flatten(-1)
        if (self.previous_state is not None and
                self.previous_action is not None):
            self.memory.add(self.previous_state,
                    self.previous_action, 0,
                    self.gb.currentReward, 0,
                    board, 0)
        self.previous_state = board
        self.previous_action = a
        
        self.update()
        return a
    
    def update(self):
        if self.step < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        (states, actions, _, reward, _, next_states,
         is_terminal) = batch
        batch_idx = 1
        #print(states[0])
        #print(states.shape)
        data = torch.from_numpy(states).type(torch.FloatTensor).cuda().view(-1, 17, 4, 4)
        #data = data.transpose(1, 2).view(-1, 17, 4, 4)
        #print(data[0].view(4,4))
        #data = data.to(device)
        #print(self.memory._next_idx)
        #print(data.shape)
        self.opt.zero_grad()
        #print(data[0])
        recon_batch, mu, logvar = self.vq(data)

        #print(recon_batch.shape, data.shape)
        #print(recon_batch.shape)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss = loss.item()
        self.opt.step()
        self.loss += loss.item() / len(data)