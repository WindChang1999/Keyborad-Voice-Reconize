import torch
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn as nn
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
        acc_list.append(now_acc*100)
        loss_list.append(loss)

        if now_acc > best_acc:
            best_acc = now_acc
            min_loss = loss
        print("Now Test accuracy = {:.4f}%".format(now_acc * 100))
        print("Now Test Loss = {}".format(loss))
        epoch_time = time.time() - epoch_time
        print("Now epoch time = {:.2f}s".format(epoch_time))

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

    best_model_state_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_state_dict)
    filename = 'loss=' + str(min_loss)[:5] + '.tar'
    torch.save(model.state_dict(),
               r'/Users/qinjingchang/Documents/GitHub/Keyborad-Voice-Reconize/Saved model/' + filename)
    print("model file is", filename)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(acc_list)
    ax[1].plot(loss_list)
    plt.show()
    return model

if __name__ == '__main__':
    total_time = time.time()
    # -----用整个FFT训练的数据集加载------
    train_excel_dir = r'/Users/qinjingchang/Documents/GitHub/Keyborad-Voice-Reconize/data/train.xlsx'
    test_excel_dir = r'/Users/qinjingchang/Documents/GitHub/Keyborad-Voice-Reconize/data/val.xlsx'
    dataset = {'train': FFTDataset(train_excel_dir), 'test': FFTDataset(test_excel_dir)}

    # 这里num_worker = 4不能用, 不然会一直有多进程的错误
    dataloader = {x: DataLoader(dataset[x], batch_size=5, shuffle=True) for x in ['train', 'test']}
    print("testset.size =", dataset['test'].sample_N)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNN.FFTCNN()
    model.load_state_dict(torch.load(r'/Users/qinjingchang/Documents/GitHub/' +
                                 r'Keyborad-Voice-Reconize/Saved model/loss=0.989.tar', map_location='cpu'))
    model = model.to(device)
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    CostFunction = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    training_model(model, CostFunction, optimizer, num_epochs=300)
    total_time = time.time() - total_time
    print("-----" * 20 + "Run Complete" + "-----" * 20)
    print("Total run time: {:.2f}s".format(total_time))