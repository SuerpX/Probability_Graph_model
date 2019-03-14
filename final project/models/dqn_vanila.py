import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

GAME_BOARD_SIZE = int(16 / 4)
class DQN_Vanila(nn.Module):
    def __init__(self):
        super(DQN_Vanila, self).__init__()
        self.input_dim = 17
        
        self.cnn_l1_a  =  nn.Sequential(
            nn.Conv2d(self.input_dim, 128, kernel_size=(1, 2), stride=1),
            nn.ReLU())
        self.cnn_l1_b  =  nn.Sequential(
            nn.Conv2d(self.input_dim, 128, kernel_size=(2, 1), stride=1),
            nn.ReLU())

        self.cnn_l2_aa  =  nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 2), stride=1),
            nn.ReLU())
        self.cnn_l2_ab  =  nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(2, 1), stride=1),
            nn.ReLU())
        self.cnn_l2_ba  =  nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 2), stride=1),
            nn.ReLU())
        self.cnn_l2_bb  =  nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(2, 1), stride=1),
            nn.ReLU())
        
        self.fc1 = nn.Sequential(
            torch.nn.Linear(7424, 256),
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
        '''
        x = self.cnn_l1(x)
        x = self.cnn_l2(x)
        '''
        '''
        torch.Size([1, 128, 4, 3])
        torch.Size([1, 128, 3, 4])
        torch.Size([1, 128, 4, 2])
        torch.Size([1, 128, 3, 3])
        torch.Size([1, 128, 3, 3])
        torch.Size([1, 128, 2, 4])
        '''
        x_a = self.cnn_l1_a(x)
        x_b = self.cnn_l1_b(x)

        x_aa = self.cnn_l2_aa(x_a)
        x_ab = self.cnn_l2_ab(x_a)
        x_ba = self.cnn_l2_ba(x_b)
        x_bb = self.cnn_l2_bb(x_b)
        '''
        print(x_a.shape)
        print(x_b.shape)
        print(x_aa.shape)
        print(x_ab.shape)
        print(x_ba.shape)
        print(x_bb.shape)
        '''

        x_a = x_a.view(-1, 128 * 4 * 3)
        x_b = x_b.view(-1, 128 * 3 * 4)
        x_aa = x_aa.view(-1, 128 * 4 * 2)
        x_ab = x_ab.view(-1, 128 * 3 * 3)
        x_ba = x_ba.view(-1, 128 * 3 * 3)
        x_bb = x_bb.view(-1, 128 * 2 * 4)

        x = torch.cat((x_a, x_b, x_aa, x_ab,x_ba, x_bb,), dim = 1)

        x = self.fc1(x)
        q_value = self.fc2(x)
        #q_value = self.fc3(x)

        return q_value
    def loss_function(self, target_q_values, q_values):

    	dqn_loss = self.dqn_loss(q_values,target_q_values)
    
    	return dqn_loss