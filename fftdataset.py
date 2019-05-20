import xlrd
import torch
from torch.utils.data import Dataset

# 只分类了30个键
def KeytypeToTarget(Keytype):
    # total_class_n = 17
    # Target = [0] * total_class_n
    MappingDict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                   'a': 10, 'b': 11, 'c': 12, 'd': 13, 'space': 14, 'back': 15, 'enter': 16}
    return MappingDict[Keytype]


class FFTDataset(Dataset):
    '''
    不同声音信号的FFT数据集
    '''

    def __init__(self, Excelfile):
        '''
        :param Excelfile: excel data root direction
        '''
        self.excel_dir = Excelfile
        self.workbook = xlrd.open_workbook(self.excel_dir)
        self.sheet = self.workbook.sheet_by_index(0)
        self.FFT_N = self.sheet.ncols - 1                    # FFT的长度
        self.sample_N = self.sheet.nrows                     # 样本数

    def __len__(self):
        '''override, return dataset length'''
        return self.sample_N

    def __getitem__(self, index):
        '''返回下标为index的对象'''
        fft = torch.tensor(self.sheet.row_values(index)[1:])    # fft -- tensor of real number
        Keytype = self.sheet.row_values(index)[0]               # Keytype -- string/char
        if type(Keytype) == float:
            Keytype = str(int(Keytype))
        else:
            Keytype = str(Keytype)
        Keytype = KeytypeToTarget(Keytype)
        sample = (fft / torch.max(fft), Keytype)
        return sample



