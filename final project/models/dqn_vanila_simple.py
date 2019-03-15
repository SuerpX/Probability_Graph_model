import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

GAME_BOARD_SIZE = int(16 / 4)
class DQN_Vanila_simple(nn.Module):
    def __init__(self):
        super(DQN_Vanila_simple, self).__init__()
        self.input_dim = 17
        
        self.cnn_l1  =  nn.Sequential(
            nn.Conv2d(self.input_dim, 128, kernel_size=2, stride=1),
            nn.ReLU())

        self.cnn_l2  =  nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=2, stride=1),
            nn.ReLU())
		
        self.fc1 = nn.Sequential(
            torch.nn.Linear(128 * GAME_BOARD_SIZE, 256),
            nn.ReLU()
        )

        self.fc2  =  nn.Sequential(
            torch.nn.Linear(256, 4),
            #nn.ReLU()#,
            #nn.BatchNorm1d(100)
        )
        '''
        self.fc3  =  nn.Sequential(
            torch.nn.Linear(100, 4)#,
            #nn.Sigmoid()
            )
        '''
        self.dqn_loss = nn.MSELoss()#reduction = 'sum')

    def forward(self, x):
        x = self.cnn_l1(x)
        x = self.cnn_l2(x)

        x = x.view(-1, 128 * GAME_BOARD_SIZE)
        
        #x = x.view(-1, 17 * GAME_BOARD_SIZE)
        x = self.fc1(x)
        q_value = self.fc2(x)
        #q_value = self.fc3(x)

        return q_value
    def loss_function(self, target_q_values, q_values):

    	dqn_loss = self.dqn_loss(q_values,target_q_values)
    
    	return dqn_loss