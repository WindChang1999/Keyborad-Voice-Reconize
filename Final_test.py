from torch import device, cuda, load, tensor
from time import time
import numpy as np
import CNN
from heapq import nlargest
from Recording import Record
from scratch import Get_fft
import xlrd
import xlsxwriter


if __name__ == '__main__':
    MappingDict = {'p': 0, 'q': 1, 'b': 2, 'z': 3, 'd': 4, 't': 5 , 'm':6, 'i':7, 'k':8}
    inv_Dict = {v:k for k, v in MappingDict.items()}

    # 打开旧文件
    old_workbook = xlrd.open_workbook("testdata.xlsx")
    old_sheet = old_workbook.sheet_by_index(0)
    old_n_row = old_sheet.nrows
    old_n_col = 2
    # 建新的文件
    new_workbook = xlsxwriter.Workbook("testdata.xlsx")
    new_sheet = new_workbook.add_worksheet()
    # 复制当前已有的数据

    for i in range(old_n_row):
        for j in range(old_n_col):
            new_sheet.write(i, j, old_sheet.row_values(i)[j])

    model_path = r'Saved model/loss=1.562.tar'
    device = device("cuda:0" if cuda.is_available() else "cpu")
    model = CNN.FFTCNN()
    model.load_state_dict(load(model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()

    acc_list = [0] * 4
    testkey = input("Please input test key type:\n")
    FILENAME = 'pred'
    ##### 接受录音、截取处理成fft
    wavfile = Record(5, FILENAME)
    # wavfile = r'recordings\b1'

    since = time()
    fft, _ = Get_fft(wavfile + '.wav')
    if len(fft.shape) == 1:                             # 只截取到一个按键音
        fft = fft.reshape(1, -1)
    fft = fft[:, 5:180]

    print("epdt run time = {:.2f}s".format(time() - since))

    for x in fft:
        x = x / np.max(x)
    fft = tensor(fft).to(device).float()

    output = model(fft).cpu()
    print("output tensor =", output)
    print("*"*80)
    lagest3 = [nlargest(3, list(output[i])) for i in range(len(output))]
    # print(lagest3)
    keyclasslagest3 = []
    for i in range(len(lagest3)):
        out = list(output[i])
        las3 = lagest3[i]
        keyclasslagest3.append([inv_Dict[out.index(x)] for x in las3])

    print("Total run time = {:.2f}s".format(time()-since))
    print("Key number:", output.shape[0])
    print("Probability lagest key class:")
    for k in keyclasslagest3:
        print(k, end='\n')

    for las3 in keyclasslagest3:
        if testkey in las3:
            acc_list[las3.index(testkey)] += 1
        else:
            acc_list[3] += 1

    pred_list = [las3[0] for las3 in keyclasslagest3]

    for i in range(old_n_row, old_n_row + output.shape[0]):
        new_sheet.write(i, 0, testkey)
        new_sheet.write(i, 1, pred_list[i-old_n_row])

    new_workbook.close()

    print(acc_list)
    output_str = ''
    for k in keyclasslagest3:
        output_str += k[0]
    print("output sequence:", output_str)
