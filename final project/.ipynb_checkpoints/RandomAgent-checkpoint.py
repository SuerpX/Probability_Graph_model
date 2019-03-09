from agents import agent
import agents
from random import randint, uniform
from vae_dqn import loss_function
from common.prioritized_memory.memory import PrioritizedReplayBuffer
from copy import deepcopy
import torch
from torch import nn, optim
import time
import numpy as np
device = torch.device("cuda")
class randomAgent(agent):
    def __init__(self, gb, model = None):
        super().__init__(gb)
        #print(common.prioritized_memory.memory)
        self.memory = PrioritizedReplayBuffer(200, 0.6)
        self.previous_state = None
        self.previous_action = None
        self.step = 0
        self.vq = model
        #self.vq.train()
        self.loss = 0
    def action(self):
        self.step += 1
        a = self.actions[randint(0, 3)]
        board = deepcopy(self.gb.board)
        #print(board)
        board[board == 0] = 1
        board = np.log2(board)
        #print(board)

        if (self.previous_state is not None and
                self.previous_action is not None):
            self.memory.add(self.previous_state,
                    self.previous_action,
                    self.gb.currentReward,
                    board, 0)
        self.previous_state = board
        self.previous_action = a
        
        self.update()
        return a
    
    def update(self):
        if self.step < 16:
            return
        
        batch = self.memory.sample(16, 0.5)
        (states, actions, reward, next_states,
         is_terminal, weights, batch_idxes) = batch
        batch_idx = 1
        data = torch.from_numpy(states).type(torch.FloatTensor).cuda()
        #data = data.to(device)
        optimizer = optim.Adam(self.vq.parameters(), lr=1e-3)
        optimizer.zero_grad()
        #print(data[0])
        recon_batch, mu, logvar = self.vq(data)
        '''
        print("**************")
        print(recon_batch[0].view(4,4))
        print("--------------")
        print(data[0])
        print("**************")
        time.sleep(0.5)
        '''
        #print(recon_batch.shape, data.shape)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss = loss.item()
        optimizer.step()
        self.loss += loss.item() / len(data)