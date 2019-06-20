from torch import device, cuda, load, tensor
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import torch.nn as nn
import torch.nn.functional as F

# MappingDict = {'p': 0, 'q': 1, 'b': 2, 'z': 3, 'd': 4, 't': 5}
MappingDict = {'p': 0, 'q': 1, 'b': 2, 'z': 3, 'd': 4, 't': 5, 'm': 6, 'i': 7, 'k':8}
# MappingDict = {'p': 0, 'q': 1, 'b': 2, 'z': 3, 'd': 4, 't': 5, '8': 6, 'h': 7}
inv_Dict = {v:k for k, v in MappingDict.items()}
class_name = [str(inv_Dict[i]) for i in range(len(inv_Dict))]

class FFTCNN(nn.Module):
    def __init__(self):
        super(FFTCNN, self).__init__()
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
        # print(x[0].cpu().detach().numpy())
        plt.plot(x[0].cpu().detach().numpy()[0], c='k')                      # 经过池化层之后
        plt.xlabel('n')
        plt.title(u'MaxPool layer output')
        plt.show()
        x = F.relu(self.conv1(x))
        plt.plot(x[0].cpu().detach().numpy()[0], c='k')                      # 经过卷积层之后
        plt.xlabel('n')
        plt.title(u'Convlution layer output')
        plt.show()
        x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        plt.plot(x.cpu().detach().numpy()[0], c='k')                         # fc1
        plt.xlabel('n')
        plt.title(u'Fully connected layer 1 output')
        plt.show()
        x = F.relu(self.fc2(x))
        plt.bar(class_name, x.cpu().detach().numpy()[0])
        plt.title(u'Finally result')
        plt.show()
        return x

if __name__ == '__main__':
    # 以z键为例
    model_path = r'Saved model/loss=1.562.tar'
    device = device("cuda:0" if cuda.is_available() else "cpu")
    model = FFTCNN()
    model.load_state_dict(load(model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    wb = xlrd.open_workbook(r'data/train.xlsx')
    sheet = wb.sheet_by_index(0)
    fft = sheet.row_values(2)[1:]
    plt.plot(fft)                                       # 原始fft
    plt.xlabel('k')
    plt.ylabel('|X(k)|')
    plt.title(u'origin fft')
    plt.show()
    fft_1 = np.array(fft[5:180])
    plt.plot(fft_1)                                     # 截取第5个点到第180个点的fft
    plt.xlabel('k')
    plt.ylabel('|X(k)|')
    plt.title(u'origin fft (k=5~180)')
    plt.show()
    output = model(tensor(fft_1.reshape(1, -1)).to(device).float())
    print(output)
