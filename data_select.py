import numpy as np
import matplotlib.pyplot as plt
from scratch import Get_fft
import xlsxwriter
import os
from Recording import Record
if __name__ == '__main__':
    # plt.ion()               # 交互模式
    dir = r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\selected data'
    Keytype = input("Please input keytype:\n")
    filename = input("Please input filename:\n")
    Record_second = int(input("Please input record :\n"))

    pfilename = Record(Record_second, filename)
    fs = 44100
    fft_list = np.array(Get_fft(os.path.join(pfilename+'.wav')))
    n_sample = len(fft_list)                # 截到的样本数量
    n_fft = len(fft_list[0])                # fft 的点数
    f = np.array([fs/2/n_fft*i for i in range(n_fft)])      # 横坐标
    workbook = xlsxwriter.Workbook(os.path.join(dir, filename+'.xlsx'))
    sheet = workbook.add_worksheet()
    print(fft_list.shape)

    cnt = 0
    for fft in fft_list:
        plt.figure()
        plt.plot(f, fft)
        plt.show()
        # ch = msvcrt.getch()
        ch = input()
        if ch == '':
            print('Saved')
            for j in range(1, n_fft+1):
                sheet.write(cnt, j, fft[j-1])         # 数据写入xlsx
            sheet.write(cnt, 0, Keytype)
            cnt += 1
        else:
            print('Discard')
    workbook.close()
    print("Run Completed")