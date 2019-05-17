import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FFTnet(nn.Module):
    '''
    四层的神经网络, 前3层用Relu激活, 最后一层用Sigmoid
    '''
    def __init__(self, size):
        '''
        :param size: 每层神经元的个数
        '''
        self.size = size
        self.input = nn.Linear(size[0], size[1])
        self.hidden1 = nn.Linear(size[1], size[2])
        self.hidden2 = nn.Linear(size[2], size[3])
        self.output = nn.Linear(size[3], size[4])

    def forward(self, x):
        x = F.relu(self.input(x))        # 1
        x = F.relu(self.hidden1(x))      # 2
        x = F.relu(self.hidden2(x))      # 3
        x = F.sigmoid(self.output(x))    # 最后输出用sigmoid限制在 0～1
        return x


