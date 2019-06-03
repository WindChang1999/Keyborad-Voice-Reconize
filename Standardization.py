import xlrd
import numpy as np

MappingDict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
               'a': 10, 'b': 11, 'c': 12, 'd': 13, 'space': 14, 'back': 15, 'enter': 16,
               'alt': 17, 'e': 18, 'f': 19, 'fn': 20, 'g': 21, 'q': 22, 'r': 23, 't': 24, 'v': 25,
               'w': 26}

excel_dir = r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\data\train.xlsx'
workbook = xlrd.open_workbook(excel_dir)
sheet = workbook.sheet_by_index(0)
total_col = sheet.ncols
data = [[]]*len(MappingDict)

for i in range(sheet.nrows):
    row = sheet.row_values(i)
    Keytype = row[0]
    if type(Keytype) == float:
        Keytype = str(int(Keytype))
    else:
        Keytype = str(Keytype)
    fft = row[1:]
    data[MappingDict[Keytype]].append(fft)

data = [ np.array(x) for x in data ]
mean = np.array([ list(np.mean(x, axis=0)) for x in data ])
mean = np.mean(mean, axis=0)

var = [ list(np.var(x, axis=0)) for x in data ]
std = np.sqrt(np.sum(var, axis=0))
print("mean=", list(mean))
print("std=", list(std))
print(len(mean), len(std))