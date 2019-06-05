from torch import device, cuda, load, tensor
from time import time
import CNN
from heapq import nlargest
from Recording import Record
from scratch import Get_fft

if __name__ == '__main__':
    # MappingDict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    #                'a': 10, 'b': 11, 'c': 12, 'd': 13, 'space': 14, 'back': 15, 'enter': 16,
    #                'alt': 17, 'e': 18, 'f': 19, 'fn': 20, 'g': 21, 'q': 22, 'r': 23, 't': 24, 'v': 25,
    #                'w': 26}

    MappingDict = {'p': 0, 'q': 1, 'b': 2, 'z': 3, 'd': 4, 'j': 5, '8': 6, 'h': 7, 't': 8}
    inv_Dict = {v:k for k, v in MappingDict.items()}

    # 输入模型的路径去加载模型
    # print("Please input model path:")
    model_path = r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\Saved model\loss=9.396.tar'
    device = device("cuda:0" if cuda.is_available() else "cpu")
    model = CNN.FFTCNN()
    model.load_state_dict(load(model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()

    FILENAME = 'pred'
    ##### 接受录音、截取处理成fft
    wavfile = Record(5, FILENAME)

    since = time()
    fft, _ = Get_fft(wavfile + '.wav')
    # fft = fft[:, 5:180]
    fft = tensor(fft).to(device).float()
    print(fft.shape)

    output = model(fft).cpu()
    print("output tensor =", output)
    print("*"*60)
    # print("Output tensor =", output)
    lagest3 = [nlargest(3, list(output[i])) for i in range(len(output))]
    # print(lagest3)
    keyclasslagest3 = []
    for i in range(len(lagest3)):
        out = list(output[i])
        las3 = lagest3[i]
        keyclasslagest3.append([inv_Dict[out.index(x)] for x in las3])
    print("Probability lagest key class:", keyclasslagest3)
    print("Total run time = {:.2f}s".format(time()-since))