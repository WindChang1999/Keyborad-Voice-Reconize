import torch
import network
import CNN
from fftdataset import FFTDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# MappingDict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
#                    'a': 10, 'b': 11, 'c': 12, 'd': 13, 'space': 14, 'back': 15, 'enter': 16,
#                'alt': 17, 'e': 18, 'f': 19, 'fn': 20, 'g': 21, 'q':22, 'r':23, 't':24, 'v':25,
#                'w':26 }

# MappingDict = {'p1': 0, 'p2': 1, 'q1': 2, 'q2': 3, 'b1': 4, 'b2': 5, 'z1':6, 'z2':7, 'd1':8, 'd2':9}
MappingDict = {'p1': 0, 'p2': 0, 'q1': 1, 'q2': 1, 'b1': 2, 'b2': 2, 'z1': 3, 'z2': 3, 'd1': 4, 'd2': 4}

def test(model):
    model.eval()
    torch.set_grad_enabled(False)
    cnt = 0
    acclist = [0]*MappingDict.__len__()
    class_sample_N = [0]*MappingDict.__len__()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # inputs.shape = [5, 325]
        outputs = model(inputs)
        # preds_index 储存outputs中每一行最大值的index
        max_, preds_index = torch.max(outputs, 1)
        # int可以和只有一个元素的tensor相加
        # cnt += torch.sum(labels == preds_index)
        # print(outputs)

        for i in range(5):
            if preds_index[i] == labels[i]:
                acclist[preds_index[i]] += 1
            class_sample_N[labels[i]] += 1

        # print(outputs)
        # print("labels =", labels)
        # print("preds_index =", preds_index)
        # print("preds1 =", preds1)

    print(acclist)
    print(class_sample_N)
    now_acc = 1.0 * sum(acclist) / dataset.sample_N
    return now_acc, acclist, class_sample_N

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = FFTDataset(r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\data\test1.xlsx')
dataloader = DataLoader(dataset, batch_size=5, shuffle=True, drop_last=True)


device = torch.device("cpu")
sizelist = [dataset.FFT_N, 250, 150, 27]
# model = network.FFTnet(sizelist)
model = CNN.FFTCNN()
model = model.to(device)
print(model)
model.load_state_dict(torch.load(r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\Saved model\loss=3.930.tar'))
model.eval()


inv_Dict = {v:k for k,v in MappingDict.items()}

acc, acclist, class_sample_N = test(model)

print(sum(class_sample_N))
acclist = [100 * acclist[i] / class_sample_N[i] for i in range(len(MappingDict) // 2)]
class_name = [inv_Dict[i] for i in range(len(inv_Dict))]

print("acc = {:.4f}%".format(acc*100))
print("acclist =")

print(acclist)
print(class_name)

plt.figure()
plt.bar(class_name, acclist)
plt.xlabel("Key categorical")
plt.ylabel("Accuracy(%)")
plt.show()




