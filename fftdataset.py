import xlrd
import torch
from torch.utils.data import Dataset
import numpy as np

# 只分类了30个键
def KeytypeToTarget(Keytype):
    # MappingDict = {'p': 0, 'q': 1, 'b': 2, 'z': 3, 'd': 4, 'j': 5, '8': 6, 'h': 7, 't': 8}
    # MappingDict = {'p': 0, 'q': 1, 'b': 2, 'z': 3, 'd': 4, 't': 5, '8': 6, 'h':7}
    MappingDict = {'p': 0, 'q': 1, 'b': 2, 'z': 3, 'd': 4, 't': 5}
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
        sample = (fft[5:180], Keytype)
        return sample



