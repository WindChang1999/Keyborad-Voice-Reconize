import torch
import torch.nn as nn
import torch.nn.functional as F

class FFTCNN(nn.Module):
    def __init__(self):
        super(FFTCNN, self).__init__()
        # # max_pool1d: 325 - (kernel_size - 1) / 4 = 322 / 4 = 81
        # self.maxpool = nn.MaxPool1d(4)
        # # conv1, 81 - （kernel_size - 1） = 77
        # self.conv1 = nn.Conv1d(1, 1, 5)
        # self.fc1 = nn.Linear(77, 60)
        # self.fc2 = nn.Linear(60, 5)

        # # max_pool1d: 325 - (kernel_size - 1) / 4 = 322 / 4 = 81
        # self.maxpool1 = nn.MaxPool1d(4)
        # # conv1, 81 - （kernel_size - 1） = 77
        # self.conv1 = nn.Conv1d(1, 1, 5)
        # # conv2, 77 - (kernel_size - 1) = 73
        # self.conv2 = nn.Conv1d(1, 1, 5)
        # # maxpool2, 73 - (2 - 1) / 2 = 36
        # self.maxpool2 = nn.MaxPool1d(2)
        # self.fc1 = nn.Linear(36, 25)
        # self.fc2 = nn.Linear(25, 9)

        # 只用0.3k ~ 12k的数据做计算
        # 175 - 2 / 3 = 58
        self.maxpool = nn.MaxPool1d(3)
        # 58 - 3 = 55
        self.conv1 = nn.Conv1d(1, 1, 4)
        self.fc1 = nn.Linear(55, 45)
        self.fc2 = nn.Linear(45, 9)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.maxpool(x)
        x = F.relu(self.conv1(x))
        x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = x.unsqueeze(1)
        # x = self.maxpool1(x)
        # x = F.relu(self.conv1(x))
        # x = x.squeeze(1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))

        # x = x.unsqueeze(1)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.maxpool(x)
        # x = x.squeeze(1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return x