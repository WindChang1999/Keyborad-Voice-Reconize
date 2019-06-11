from torch import device, cuda, load, tensor
from time import time
import CNN
from heapq import nlargest
from Recording import Record
from scratch import Get_fft

if __name__ == '__main__':
    MappingDict = {'p': 0, 'q': 1, 'b': 2, 'z': 3, 'd': 4, 't': 5, '8': 6, 'h':7}
    inv_Dict = {v:k for k, v in MappingDict.items()}

    model_path = r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\Saved model\loss=5.953.tar'
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
    if len(fft.shape) == 1:                             # 只截取到一个按键音
        fft = fft.reshape(1, -1)
    fft = fft[:, 5:180]
    fft = tensor(fft).to(device).float()
    print(fft.shape)

    output = model(fft).cpu()
    print("output tensor =", output)
    print("*"*60)

    lagest3 = [nlargest(3, list(output[i])) for i in range(len(output))]
    # print(lagest3)
    keyclasslagest3 = []
    for i in range(len(lagest3)):
        out = list(output[i])
        las3 = lagest3[i]
        keyclasslagest3.append([inv_Dict[out.index(x)] for x in las3])
    print("Probability lagest key class:", keyclasslagest3)
    print("Total run time = {:.2f}s".format(time()-since))