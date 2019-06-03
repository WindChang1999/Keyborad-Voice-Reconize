import torch
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn as nn
import network
from fftdataset import FFTDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import CNN
import copy
import time



def train(model, CostFunction, optimizer):
    model.train()
    torch.set_grad_enabled(True)
    loss_sum = 0.0
    cnt = 0
    for inputs, labels in dataloader['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = CostFunction(outputs, labels)
        loss_sum += loss.cpu().item()
        max_, preds_index = torch.max(outputs, 1)
        # int可以和只有一个元素的tensor相加
        cnt += torch.sum(labels == preds_index)
        loss.backward()
        optimizer.step()
    now_acc = 1.0 * cnt.item() / dataset['train'].sample_N
    print('Now Train Loss ={}'.format(loss_sum))
    print('Now Train accuracy ={:.4f}%'.format(now_acc*100))
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
    min_loss = 100000
    acc_list = []
    loss_list = []
    best_model_state_dict = copy.deepcopy(model.state_dict())
    cnt = 0
    for epoch in range(num_epochs):
        # scheduler.step()
        epoch_time = time.time()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 60)
        model = train(model, CostFunction, optimizer)
        now_acc, loss = test(model, CostFunction)
        acc_list.append(now_acc)
        loss_list.append(loss)

        if min_loss > loss:
            best_acc = now_acc
            min_loss = loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
        print("Now Test accuracy = {:.4f}%".format(now_acc * 100))
        print("Now Test Loss = {}".format(loss))
        epoch_time = time.time() - epoch_time
        print("Now epoch time = {}".format(epoch_time))

        cnt += 1
        if cnt >= 100 and cnt % 50 == 0:
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
    plt.figure()
    plt.plot(acc_list)
    plt.plot(loss_list)
    plt.show()
    return model

if __name__ == '__main__':
    total_time = time.time()
    # -----用整个FFT训练的数据集加载------
    train_excel_dir = r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\data\train1.xlsx'
    test_excel_dir = r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\data\test1.xlsx'
    dataset = {'train': FFTDataset(train_excel_dir), 'test': FFTDataset(test_excel_dir)}
    # sizelist = [dataset['train'].FFT_N, 250, 150, 27]

    # -----用FFT_peak训练的数据集加载------
    # train_excel_dir = r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\data\train_peak_arg.xls'
    # test_excel_dir = r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\data\val_peak_arg.xls'
    # dataset = {'train': FFTPeakDataset(train_excel_dir), 'test': FFTPeakDataset(test_excel_dir)}
    # sizelist_fftpeak = [dataset['train'].FFT_N, 50, 50, 27]

    # print(dataset['train'].__getitem__(5))
    # print(dataset['test'].__getitem__(5))

    # 这里num_worker = 4不能用, 不然会一直有多进程的错误
    dataloader = {x: DataLoader(dataset[x], batch_size=5, shuffle=True) for x in ['train', 'test']}
    print("testset.size =", dataset['test'].sample_N)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # model = network.FFTnet(sizelist_fftpeak)
    model = CNN.FFTCNN()

    model = model.to(device)
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    CostFunction = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    training_model(model, CostFunction, optimizer, num_epochs=100)
    total_time = time.time() - total_time
    print("-----" * 20 + "Run Complete" + "-----" * 20)
    print("Total run time: {}".format(total_time))