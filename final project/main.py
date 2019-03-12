import agents.agents
#from agents.MCAgent import MCAgent
from agents.RandomAgent import randomAgent
#from agents.UCTAgent import UCTAgent
from agents.testAgent import testAgent
from agents.VAE_DQNAgent import VAE_DQNAgent
from agents.DQNAgent_Vanila import DQNAgent_Vanila
#from agents.TDAgent import QLAgent, SARSAAgent
from gameboard import gameboard
import time
import os
from threading import Thread, Lock, Semaphore
from time import sleep
import shutil
import torch
from utility import normalization, oneHotMap, reverseOneHotMap
from models.vae_my import VAE_DQN
from models.vae_dqn_cnn import VAE_DQN_CNN
from models.vae_cnn import VAE_CNN
from models.dqn_vanila import DQN_Vanila
import numpy as np
import models.vae_vanila
from torch import nn, optim
import tensorflow as tf
#torch.set_printoptions(precision = 2)
np.set_printoptions(precision = 4, suppress = True)

def main():
    
    #DQN

    tscore = 0
    #vae_dqn = VAE_DQN_CNN().cuda()
    dqn = DQN_Vanila().cuda()
    
    optimizer = optim.Adam(dqn.parameters(), lr=1e-4)
    '''
    for name, param in vae_dqn[0].named_parameters():
        print(name)
    '''
    agent = DQNAgent_Vanila(dqn, optimizer)
    acc = 0
    for i in range(100000):
        agent.enableLearning()
        dqn.train()
        gb = gameboard(4, isPrint = False)
        agent.play(gb)
        tscore += gb.score
        #print(agent.test_q)
        #input()
        if i % 100 == 0:
            dqn.eval()
            agent.disableLearning()
            test_score = 0
            test_num = 30
            for _ in range(test_num):
                gb = gameboard(4, isPrint = False)
                agent.play(gb)
                test_score += gb.score
            print("\ntest score: {}".format(test_score / test_num))
            
        print("\repoch: {}, loss: {}, step: {}".format(i, agent.loss / agent.step, agent.step), end = '')
    print(tscore/1000)
    """

    #VAE_DQN
    tscore = 0
    #vae_dqn = VAE_DQN_CNN().cuda()
    vae = VAE_CNN().cuda()
    vae_dqn = VAE_DQN_CNN(vae.encoder).cuda()
    vae_dqn = [vae, vae_dqn]
    
    #optimizer = optim.Adam(vae_dqn.parameters(), lr=1e-4)
    optimizers = [optim.Adam(vae_dqn[0].parameters(), lr=1e-4),optim.Adam(vae_dqn[1].parameters(), lr=1e-4)]
    '''
    for name, param in vae_dqn[0].named_parameters():
        print(name)
    '''
    agent = VAE_DQNAgent(vae_dqn, optimizers)
    acc = 0
    for i in range(100000):
        agent.enableLearning()
        vae_dqn[0].train()
        vae_dqn[1].train()
        gb = gameboard(4, isPrint = False)
        agent.play(gb)
        tscore += gb.score
        #print(agent.test_q)
        #input()
        if i % 100 == 0:
            vae_dqn[0].eval()
            vae_dqn[1].eval()
            test_score = 0
            agent.disableLearning()
            test_num = 30
            for _ in range(test_num):
                gb = gameboard(4, isPrint = False)
                agent.play(gb)
                test_score += gb.score
            print("\ntest score: {}".format(test_score / test_num))

        print("\repoch: {}, loss_vae: {}, loss_dqn: {}, step: {}".format(i, agent.loss_vae / agent.step, agent.loss_dqn / agent.step, agent.step), end = '')
    print(tscore/1000)
    """
    """
    #VAE
    tscore = 0
    #vae_dqn = VAE_DQN().cuda()
    vae_dqn = VAE_CNN().cuda()
    #vae_dqn = vae.VAE().cuda()
    testa = np.array(
    	[[64., 32., 2 ** 14, 16.],
        [16., 2., 4., 8.],
        [1024., 2 ** 15, 32., 2 ** 16],
        [4096., 8., 16., 0.]])
    testb = oneHotMap(testa)
    print(testb.shape)
    
    optimizer = optim.Adam(vae_dqn.parameters(), lr=1e-4)
    agent = randomAgent(vae_dqn, optimizer)
    acc = 0
    #summary_writer = tf.summary.FileWriter
    #tf.summary.scalar("train_loss", train_loss)
    for i in range(100000):
        vae_dqn.train()
        gb = gameboard(4, isPrint = False)
        agent.play(gb)
        tscore += gb.score
        if i % 100 == 0:
            acc = 0
            r, _, _ = vae_dqn(torch.from_numpy(testb).type(torch.float).cuda().view(-1, 17, 4, 4))
            print()
            print(testa)
            print("----------------------")
            print(reverseOneHotMap(r.data.cpu().numpy()))
        vae_dqn.eval()
        ta = testAgent(vae_dqn)
        gb = gameboard(4, isPrint = False)
        ta.play(gb)
        acc += ta.acc
        print("epoch: {}, loss: {}, acc: {}".format(i, agent.loss / agent.step, acc / (i % 100 + 1)), end = '\r')
    print(tscore/1000)
    """
main()
