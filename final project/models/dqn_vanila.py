import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

GAME_BOARD_SIZE = int(16 / 4)
class DQN_Vanila(nn.Module):
    def __init__(self):
        super(DQN_Vanila, self).__init__()
        self.input_dim = 17
        
        self.cnn_l1  =  nn.Sequential(
            nn.Conv2d(self.input_dim, 40, kernel_size=2, stride=1),
            nn.ReLU())

        self.cnn_l2  =  nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=2, stride=1),
            nn.ReLU())
		
        self.fc1 = nn.Sequential(
            torch.nn.Linear(40 * GAME_BOARD_SIZE, 150),
            nn.ReLU()
        )

        self.fc2  =  nn.Sequential(
            torch.nn.Linear(150, 75),
            nn.ReLU()#,
            #nn.BatchNorm1d(100)
        )

        self.fc3  =  nn.Sequential(
            torch.nn.Linear(75, 4)#,
            #nn.Sigmoid()
            )
        self.dqn_loss = nn.SmoothL1Loss(reduction = 'sum')

    def forward(self, x):
        
        x = self.cnn_l1(x)
        x = self.cnn_l2(x)

        x = x.view(-1, 40 * GAME_BOARD_SIZE)
        
        #x = x.view(-1, 17 * GAME_BOARD_SIZE)
        x = self.fc1(x)
        x = self.fc2(x)
        q_value = self.fc3(x)

        return q_value
    def loss_function(self, target_q_values, q_values):

    	dqn_loss = self.dqn_loss(q_values,target_q_values)
    
    	return dqn_loss