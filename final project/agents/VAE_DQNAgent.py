from agents.agents import agent
from random import randint, uniform, random
from memory import ReplayBuffer
from copy import deepcopy
import torch
from torch import nn, optim
import time
import numpy as np
from utility import normalization, oneHotMap, reverseOneHotMap, loss_function
from baselines.common.schedules import LinearSchedule

device = torch.device("cuda")
class VAE_DQNAgent(agent):
    def __init__(self, model, opt, learning = True):
        super().__init__()
        self.memory = ReplayBuffer(500)
        self.previous_state = None
        self.previous_action = None
        self.step = 0
        self.model = model
        self.loss = 0
        self.batch_size = 64
        self.opt = opt
        self.epsilon_schedule = LinearSchedule(10000,
                                               initial_p=0.99,
                                               final_p=0.01)
        self.learning = learning

    def should_explore(self):
        self.epsilon = self.epsilon_schedule.value(self.step)
        return random() < self.epsilon

    def action(self):
        if self.learning:
            self.step += 1
        
        board = deepcopy(self.gb.board)
        board = oneHotMap(board)

        if self.learning and self.should_explore():
            q_values = None
            action = randint(0, 3)
            choice = self.actions[action]
        else:
            state = torch.from_numpy(board).type(torch.FloatTensor).cuda().view(-1, 17, 4, 4)
            action, q_values = self.predict(state)
            choice = self.actions[action]
        if self.learning:
            if (self.previous_state is not None and
                    self.previous_action is not None):
                self.memory.add(self.previous_state,
                        self.previous_action,
                        self.gb.currentReward,
                        board, 0)

        self.previous_state = board
        self.previous_action = action

        if self.learning:
            self.update()
        return choice
    def enableLearning(self):
        self.learning = True
    def disableLearning(self):
        self.learning = False
    def update(self):
        if self.step < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        (states, actions, reward, next_states,
         is_terminal) = batch
        batch_idx = 1

        terminal = torch.tensor(is_terminal).type(torch.cuda.FloatTensor)
        reward = torch.tensor(reward).type(torch.cuda.FloatTensor)
        states = torch.from_numpy(states).type(torch.FloatTensor).cuda().view(-1, 17, 4, 4)
        next_states = torch.from_numpy(next_states).type(torch.FloatTensor).cuda().view(-1, 17, 4, 4)
        # Current Q Values
        q_actions, q_values, recon_batch, mu, logvar = self.predict_batch(states)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)
        #print(actions)
        #print(q_values)
        q_values = q_values[batch_index, actions]
        #print(q_values)

        # Calculate target
        q_actions_next, q_values_next, _, _, _ = self.predict_batch(next_states)
        q_max = q_values_next.max(1)[0].detach()
        q_max = (1 - terminal) * q_max

        q_target = reward + 0.99 * q_max

        self.opt.zero_grad()
        loss = self.model.loss_function(recon_batch, states, mu, logvar, q_target, q_values)
        loss.backward()
        train_loss = loss.item()
        self.opt.step()
        self.loss += loss.item() / len(states)


    def predict_batch(self, input):
        input = input
        q_values, x_hat, mu, logvar = self.model(input)
        values, q_actions = q_values.max(1)
        return q_actions, q_values, x_hat, mu, logvar

    def predict(self, input):
        q_values, x_hat, mu, logvar = self.model(input)
        action = torch.argmax(q_values)
        return action.item(), q_values