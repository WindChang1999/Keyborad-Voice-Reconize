import torch
import torch.optim as optim
import torch.nn as nn
import network
from fftdataset import FFTDataset
from torch.utils.data import DataLoader
import copy
import time



def train(model, CostFunction, optimizer):
    model.train()
    torch.set_grad_enabled(True)
    for inputs, labels in dataloader['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = CostFunction(outputs, labels)
        loss.backward()
        optimizer.step()
    return model


def test(model, CostFunction):
    model.eval()
    torch.set_grad_enabled(False)
    cnt = 0
    loss_sum = 0.0
    for inputs, labels in dataloader['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = CostFunction(outputs, labels)
        loss_sum += loss.cpu().item()
        # preds_index 储存outputs中每一行最大值的index
        max_, preds_index = torch.max(outputs, 1)
        # int可以和只有一个元素的tensor相加
        cnt += torch.sum(labels == preds_index)
        # print(outputs)

        # for i in range(5):
        #     outputs[i][preds_index[i]] = 0
        # max_, preds1 = torch.max(outputs, 1)
        # cnt += torch.sum(labels == preds1)

        # print(outputs)
        # print("labels =", labels)
        # print("preds_index =", preds_index)
        # print("preds1 =", preds1)


    now_acc = 1.0 * cnt.item() / dataset['test'].sample_N
    return now_acc, loss_sum


def training_model(model, CostFunction, optimizer, num_epochs=1000):
    best_acc = 0.0
    min_loss = 100
    acc_list = []
    loss_list = []
    best_model_state_dict = copy.deepcopy(model.state_dict())
    cnt = 0
    for epoch in range(num_epochs):
        epoch_time = time.time()
        model = train(model, CostFunction, optimizer)
        now_acc, loss = test(model, CostFunction)
        acc_list.append(now_acc)
        loss_list.append(loss)

        if min_loss > loss:
            best_acc = now_acc
            min_loss = loss
            best_model_state_dict = copy.deepcopy(model.state_dict())

        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 60)
        print("Now Test accuracy = {:.4f}%".format(now_acc * 100))
        print("Now Loss = {}".format(loss))
        epoch_time = time.time() - epoch_time
        print("Now epoch time = {}".format(epoch_time))

        cnt += 1
        if cnt > 500 and cnt % 200 == 0:
            print('-'*60)
            print("continue train? (Y/N)")
            s = input()
            if s == 'N':
                break

    print('*' * 60)
    print("Training Complete")
    print("Best accuracy = {}%".format(best_acc * 100))
    model.load_state_dict(best_model_state_dict)
    torch.save(model.state_dict(),
               r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\Saved model\\' + 'loss=' + str(min_loss)[:5] + '.tar')
    return model

if __name__ == '__main__':
    total_time = time.time()
    train_excel_dir = r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\data\train.xlsx'
    test_excel_dir = r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\data\val.xlsx'
    dataset = {'train': FFTDataset(train_excel_dir), 'test': FFTDataset(test_excel_dir)}

    # 这里num_worker = 4不能用, 不然会一直有多进程的错误
    dataloader = {x: DataLoader(dataset[x], batch_size=5, shuffle=True) for x in ['train', 'test']}

    print("testset.size =", dataset['test'].sample_N)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sizelist = [dataset['train'].FFT_N, 200, 100, 17]
    model = network.FFTnet(sizelist)
    model = model.to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    CostFunction = nn.CrossEntropyLoss()
    training_model(model, CostFunction, optimizer)
    total_time = time.time() - total_time
    print("-----" * 20 + "Run Complete" + "-----" * 20)
    print("Total run time: {}".format(total_time))