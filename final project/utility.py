import torch
import numpy as np
from torch.nn import functional as F
def oneHotMap(state):
    state = np.copy(state)
    numberList = 2 ** np.array(list(range(0, 17)))
    state[state == 0] = 1
    target_state = np.zeros((len(numberList), 4, 4))
    for i, num in enumerate(numberList):
        target_state[i][(state == num)] = 1
    #print(target_state.shape)
    return target_state

def reverseOneHotMap(state):
    target_state = np.copy(state)
    target_state = 2 ** np.argmax(np.reshape(target_state, (17,4,4)), axis = 0)
    target_state[target_state == 1] = 0
    return target_state

def normalization(state):
    state = np.copy(state)
    state[state == 0] = 1
    return np.log2(state)/16


# Reconstruction + KL divergence losses summed over all elements and batch
GAME_BOARD_SIZE = 16
def loss_function(recon_x, x, mu, logvar):
    #print(recon_x[0],x.view(-1, GAME_BOARD_SIZE)[0])
    #print(recon_x.shape)
    #print(x.shape)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    #print("loss",0.001*BCE.data, KLD.data)
    #input()
    return BCE + KLD
'''
testb = np.array(
    [[0., 0., 0., 0.],
    [16., 0., 0., 0.],
    [2., 2., 32., 0.],
    [4., 8., 16., 4.]])
print(oneHotMap(testb))
print(reverseOneHotMap(oneHotMap(testb)))
'''