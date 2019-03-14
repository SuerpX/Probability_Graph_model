from agents.agents import agent
from random import randint, uniform
import random
from memory import ReplayBuffer
from copy import deepcopy
import torch
from torch import nn, optim
import time
import numpy as np
from utility import normalization, oneHotMap, reverseOneHotMap, loss_function
from baselines.common.schedules import LinearSchedule
from gameboard import gameboard

device = torch.device("cuda")
class DQNAgent_Vanila(agent):
    def __init__(self, model, opt, learning = True):
        super().__init__()
        self.memory = ReplayBuffer(6000)
        self.previous_state = None
        self.previous_action = None
        self.previous_legal_actions = None
        self.step = 0
        self.model = model
        self.opt = opt
        self.loss = 0
        self.batch_size = 256
        self.test_q = 0
        #self.test_q = 0
        self.epsilon_schedule = LinearSchedule(1000000,
                                               initial_p=0.99,
                                               final_p=0.01)
        self.learning = learning

    def should_explore(self):
        self.epsilon = self.epsilon_schedule.value(self.step)
        return random.random() < self.epsilon

    def action(self):
        if self.learning:
            self.step += 1
        
        legalActions = self.legal_actions(deepcopy(self.gb.board))
        board = deepcopy(self.gb.board)
        board = oneHotMap(board)

        if self.learning and self.should_explore():
            q_values = None
            action = random.choice(legalActions)
            choice = self.actions[action]
        else:
            #mark
            state = torch.from_numpy(board).type(torch.FloatTensor).cuda().view(-1, 17, 4, 4)
            action, q_values = self.predict(state, legalActions)
            choice = self.actions[action]
        if self.learning:
            reward = self.gb.currentReward
            if reward != 0:
                reward = np.log2(reward)
            if (self.previous_state is not None and
                    self.previous_action is not None):
                self.memory.add(self.previous_state,
                        self.previous_action, self.previous_legal_actions,
                        reward, legalActions,
                        board, 0)

        self.previous_state = board
        self.previous_action = action
        self.previous_legal_actions = legalActions

        if self.learning:
            self.update()
        return choice

    def enableLearning(self):
        self.learning = True
        self.reset()

    def disableLearning(self):
        self.learning = False
    def end_episode(self):
        if not self.learning:
            return
        #print(self.gb.board)
        board = deepcopy(self.gb.board)
        board = oneHotMap(board)

        #legalActions = self.legal_actions(deepcopy(self.gb.board))
        #print(legalActions)
        self.memory.add(self.previous_state,
        self.previous_action, self.previous_legal_actions,
        self.gb.currentReward, [],
        board, 1)
        self.reset()

    def reset(self):
        self.previous_state = None
        self.previous_action = None
        self.previous_legal_actions = None

    def update(self):
        if self.step < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        (states, actions, legal_actions, reward, next_legal_actions, next_states,
         is_terminal) = batch

        terminal = torch.tensor(is_terminal).type(torch.cuda.FloatTensor)
        reward = torch.tensor(reward).type(torch.cuda.FloatTensor)
        states = torch.from_numpy(states).type(torch.FloatTensor).cuda().view(-1, 17, 4, 4)
        next_states = torch.from_numpy(next_states).type(torch.FloatTensor).cuda().view(-1, 17, 4, 4)
        # Current Q Values

        _, q_values = self.predict_batch(states)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)
        #print(actions)
        #print(q_values)
        self.test_q = terminal
        q_values = q_values[batch_index, actions]
        #print(q_values)
        # Calculate target
        q_actions_next, q_values_next = self.predict_batch(next_states, legalActions = next_legal_actions)
        #print(q_values_next)
        q_max = q_values_next.max(1)[0].detach()

        q_max = (1 - terminal) * q_max
        # if sum(terminal == 1) > 0:
        #     print(reward)
        #     print( (terminal == 1).nonzero())
        #     print(terminal)
        #     print(next_legal_actions)
        #     print(q_max)
        #     input()
        q_target = reward + 0.99 * q_max
        self.opt.zero_grad()
        loss = self.model.loss_function(q_target, q_values)

        loss.backward()
        
        self.opt.step()

        #train_loss = loss_vae.item() + loss_dqn.item()


        self.loss += loss.item() / len(states)


    def predict_batch(self, input, legalActions = None):
        input = input
        #print(legalActions)

        q_values = self.model(input)
        if legalActions is None:
            values, q_actions = q_values.max(1)
        else:
            isNotlegal = True
            while isNotlegal:
                isNotlegal = False
                values, q_actions = q_values.max(1)
                #print(q_values)
                #print(values)
                #print(q_actions)
                for i, action in enumerate(q_actions):
                    #print(legalActions[i])
                    if len(legalActions[i]) == 0:
                        continue

                    if action.item() not in legalActions[i]:
                        isNotlegal = True
                        # print(i)
                        # print(action.item())
                        # print(q_values)
                        q_values[i, action] = -1
                #         print(q_values)
                # print("*********************")

        return q_actions, q_values

    def predict(self, input, legalActions):
        q_values = self.model(input)
        isNotlegal = True
        while isNotlegal:
            #mark
            action = torch.argmax(q_values)
            if action in legalActions:
                isNotlegal = False
            else:
                q_values[0, action] = -1

        return action.item(), q_values

    def legal_actions(self, copy_gb):
        legalActions = []
        for i in range(4):
            try_gb = gameboard(4, deepcopy(copy_gb))
            changed = try_gb.takeAction(self.actions[i])
            if changed:
                legalActions.append(i)
        return legalActions
    '''
    def play(self, gb):
        self.gb = gb
        while not self.gb.islost:
            self.gb.takeAction(self.action())
        #print(1)
    '''