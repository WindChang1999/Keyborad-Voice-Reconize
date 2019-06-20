import torch
import CNN
from fftdataset import FFTDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# MappingDict = {'p': 0, 'q': 1, 'b': 2, 'z': 3, 'd': 4, 't': 5}
MappingDict = {'p': 0, 'q': 1, 'b': 2, 'z': 3, 'd': 4, 't': 5, 'm': 6, 'i': 7, 'k':8}
# MappingDict = {'p': 0, 'q': 1, 'b': 2, 'z': 3, 'd': 4, 't': 5, '8': 6, 'h': 7}
inv_Dict = {v:k for k, v in MappingDict.items()}

def test(model):
    model.eval()
    torch.set_grad_enabled(False)
    cnt = 0
    acclist = [0]*len(inv_Dict)
    class_sample_N = [0]*len(inv_Dict)
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
dataset = FFTDataset(r'/Users/qinjingchang/Documents/GitHub/Keyborad-Voice-Reconize/data/test.xlsx')
dataloader = DataLoader(dataset, batch_size=5, shuffle=True, drop_last=True)


device = torch.device("cpu")
# model = network.FFTnet(sizelist)
model = CNN.FFTCNN()
model = model.to(device)
print(model)
model.load_state_dict(torch.load(r'/Users/qinjingchang/Documents/GitHub/'+
                                 r'Keyborad-Voice-Reconize/Saved model/loss=1.562.tar', map_location='cpu'))
model.eval()

acc, acclist, class_sample_N = test(model)

print(sum(class_sample_N))
acclist = [100 * acclist[i] / class_sample_N[i] for i in range(len(inv_Dict))]
class_name = [str(inv_Dict[i]) for i in range(len(inv_Dict))]

print("acc = {:.4f}%".format(acc*100))
print("acclist =")

print(acclist)
print(class_name)

plt.figure()
plt.bar(class_name, acclist)
plt.xlabel("Key categorical")
plt.ylabel("Accuracy(%)")
plt.show()




