import torch
import torch.nn as nn
import torch.nn.functional as F

class FFTCNN(nn.Module):
    '''
    四层的神经网络, 前3层用Relu激活, 最后一层用Sigmoid
    '''
    def __init__(self):
        super(FFTCNN, self).__init__()
        # max_pool1d: 325 - (kernel_size - 1) / 4 = 322 / 4 = 81
        self.maxpool = nn.MaxPool1d(4)
        # conv1, 81 - （kernel_size - 1） = 77
        self.conv1 = nn.Conv1d(1, 1, 5)
        self.fc1 = nn.Linear(77, 60)
        self.fc2 = nn.Linear(60, 5)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.maxpool(x)
        x = F.relu(self.conv1(x))
        x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x