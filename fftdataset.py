import xlrd
import torch
from torch.utils.data import Dataset
import numpy as np

# 只分类了30个键
def KeytypeToTarget(Keytype):
    # MappingDict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    #                'a': 10, 'b': 11, 'c': 12, 'd': 13, 'space': 14, 'back': 15, 'enter': 16,
    #                'alt': 17, 'e': 18, 'f': 19, 'fn': 20, 'g': 21, 'q': 22, 'r': 23, 't': 24, 'v': 25,
    #                'w': 26}
    MappingDict = {'p1': 0, 'p2': 0, 'q1': 1, 'q2': 1, 'b1': 2, 'b2': 2, 'z1': 3, 'z2': 3, 'd1': 4, 'd2': 4}
    return MappingDict[Keytype]


class FFTDataset(Dataset):
    '''
    不同声音信号的FFT数据集
    '''

    def __init__(self, Excelfile, transforms=None):
        '''
        :param Excelfile: excel data root direction
        '''
        self.excel_dir = Excelfile
        self.workbook = xlrd.open_workbook(self.excel_dir)
        self.sheet = self.workbook.sheet_by_index(0)
        self.FFT_N = self.sheet.ncols - 1                    # FFT的长度
        self.sample_N = self.sheet.nrows                     # 样本数
        self.transforms = transforms

    def __len__(self):
        '''override, return dataset length'''
        return self.sample_N

    def __getitem__(self, index):
        '''返回下标为index的对象'''
        fft = torch.tensor(self.sheet.row_values(index)[1:])  # fft -- tensor of real number
        # fft = torch.tensor((np.array(fft) - np.array(mean)) / np.array(std))
        # fft = fft.float()
        Keytype = self.sheet.row_values(index)[0]               # Keytype -- string/char
        if type(Keytype) == float:
            Keytype = str(int(Keytype))
        else:
            Keytype = str(Keytype)
        Keytype = KeytypeToTarget(Keytype)
        sample = (fft, Keytype)
        return sample



