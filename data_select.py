import numpy as np
import matplotlib.pyplot as plt
from scratch import Get_fft
import xlsxwriter
import os

from Recording import Record


if __name__ == '__main__':
    plt.ion()               # 交互模式
    dir = r'/Users/qinjingchang/Documents/GitHub/Keyborad-Voice-Reconize/selected data'
    Keytype = input("Please input keytype:\n")
    filename = input("Please input filename:\n")
    Record_second = int(input("Please input record:\n"))

    pfilename = Record(Record_second, filename)
    # pfilename = r'recordings/z1'
    fs = 44100
    fft_list, t_list = np.array(Get_fft(os.path.join(pfilename+'.wav')))
    n_sample = len(fft_list)                                            # 截到的样本数量
    n_fft = 325                                            # fft 的点数
    print("fft_list = ", fft_list,
          "\nt_list = ", t_list)
    # f = np.array([fs/2/n_fft*i for i in range(n_fft)])                  # 横坐标

    workbook = xlsxwriter.Workbook(os.path.join(dir, filename+'.xlsx'))
    sheet = workbook.add_worksheet()
    print(fft_list.shape)

    cnt = 0
    for fft, t in zip(fft_list, t_list):
        f, ax = plt.subplots(2, 1)
        ax[0].plot(fft[5:180], 'black')
        # ax[0].set_ylim(0, 35)
        ax[1].plot(t, 'black')
        # ax[1].set_ylim(-1.2, 1.2)
        plt.pause(0.5)
        plt.close()
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