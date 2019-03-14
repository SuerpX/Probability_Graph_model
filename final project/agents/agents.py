from memory import ReplayBuffer
from copy import deepcopy
import torch
device = torch.device("cuda")
class agent():
    def __init__(self):#, mutex = None):
        self.gb = 0
        self.actions = ['u', 'd', 'l', 'r']
        self.score = 0
        self.rewardOfActions = []
 #       self.mutex = mutex
    def action(self):
        pass
    def play(self, gb):
        self.gb = gb
        while not self.gb.islost:
            self.gb.takeAction(self.action())
        self.end_episode()
    def end_episode(self):
        pass