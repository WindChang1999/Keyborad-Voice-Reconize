import xlrd
import torch
from torch.utils.data import Dataset

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
        self.sheet = self.workbook.sheet_by_index()[0]
        self.FFT_N = self.sheet.ncols - 1                    # FFT的长度
        self.sample_N = self.sheet.nrows                     # 样本数

    def __len__(self):
        '''override, return dataset length'''
        return self.sample_N

    def __getitem__(self, index):
        '''返回下标为index的对象'''
        fft = torch.Tensor(self.sheet.row_values(index)[:-1])    # fft -- tensor of real number
        Keytype = self.sheet.row_values(index)[-1]               # Keytype -- string/char
        sample = {'FFT': fft, 'Keytype': Keytype}
        return sample



