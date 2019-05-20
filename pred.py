import torch
import torch.optim as optim
import torch.nn as nn
import network
from fftdataset import FFTDataset
from torch.utils.data import DataLoader
import copy
import time

def test(model):
    model.eval()
    torch.set_grad_enabled(False)
    cnt = 0
    acclist = [0]*17
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        # preds_index 储存outputs中每一行最大值的index
        max_, preds_index = torch.max(outputs, 1)
        # int可以和只有一个元素的tensor相加
        # cnt += torch.sum(labels == preds_index)
        # print(outputs)

        for i in range(5):
            if preds_index[i] == labels[i]:
                acclist[preds_index[i]] += 1

        # print(outputs)
        # print("labels =", labels)
        # print("preds_index =", preds_index)
        # print("preds1 =", preds1)


    now_acc = 1.0 * sum(acclist) / dataset.sample_N
    return now_acc, acclist

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
sizelist = [325, 200, 100, 17]
model = network.FFTnet(sizelist)
model = model.to(device)
print(model)
model.load_state_dict(torch.load(r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\Saved model\loss=37.82.tar'))
model.eval()

dataset = FFTDataset(r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\data\val.xlsx')
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)


MappingDict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                   'a': 10, 'b': 11, 'c': 12, 'd': 13, 'space': 14, 'back': 15, 'enter': 16}
inv_Dict = {v:k for k,v in MappingDict.items()}

acc, acclist = test(model)
print("acc = {:.4f}%".format(acc*100))
print("acclist =")
print(acclist)
print([inv_Dict[i] for i in range(len(inv_Dict))])



