import agents
from MCAgent import MCAgent
from RandomAgent import randomAgent
from UCTAgent import UCTAgent
from TDAgent import QLAgent, SARSAAgent
from gameboard import gameboard
import time
import os
from threading import Thread, Lock, Semaphore
from time import sleep
import shutil
import torch

from vae_dqn import VAE_DQN
agentType = {"MCAgent": MCAgent}
def main():
    tscore = 0
    vae_dqn = VAE_DQN().cuda()
    testb = torch.tensor([[0., 0., 0., 0.],
        [4., 0., 0., 0.],
        [1., 1., 5., 0.],
        [2., 3., 4., 2.]], device='cuda:0')
    vae_dqn.train()
    for i in range(500):
        gb = gameboard(4, isPrint = False)
        agent = randomAgent(gb, vae_dqn)
        agent.play()
        tscore += gb.score
        print(agent.loss / agent.step)
        if i % 50 == 0:
            r, _, _ = vae_dqn(testb)
            print(r.view(4, 4))
    print(tscore/500)
    
    
main()
