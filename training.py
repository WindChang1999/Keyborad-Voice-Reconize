import torch
import torch.optim as optim
import torch.nn as nn
import network
from fftdataset import FFTDataset
from torch.utils.data import DataLoader
import copy



def train(model, CostFunction, optimizer):
    model.train()
    torch.set_grad_enabled(True)
    for samples in DataLoader['train']:
        inputs = samples['FFT'].to(device)
        labels = KeytypeToTarget(samples['Keytype']).to(device)
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
    for samples in DataLoader['test']:
        inputs = samples['FFT'].to(device)
        labels = KeytypeToTarget(samples['Keytype']).to(device)
        outputs = model(inputs)
        # preds_index 储存outputs中每一行最大值的index
        max_, preds_index = torch.max(outputs, 1)
        loss = CostFunction(outputs, labels)
        loss_sum += loss.cpu().item()

        # labels == preds_index 返回一个tensor
        # labels == tensor([5, 3 ,2])
        # preds_index == tensor([3, 3, 2])
        # labels == preds_index -> tensor([0, 1, 1])
        # int可以和只有一个元素的tensor相加
        cnt += torch.sum(labels == preds_index)
    now_acc = 1.0 * cnt.item() / dataset['train'].sample_N
    return now_acc, loss_sum


def training_model(model, CostFunction, optimizer, num_epochs=100):
    best_acc = 0.0
    min_loss = 100
    best_model_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 60)
        model = train(model, CostFunction, optimizer)
        now_acc, loss = test(model, CostFunction)

        if min_loss > loss:
            best_acc = now_acc
            min_loss = loss
            best_model_state_dict = copy.deepcopy(model.state_dict())

        print("Now Test accuracy = {:.4f}%".format(now_acc * 100))
        print("Now Loss = {}".format(loss))

    print('*' * 60)
    print("Training Complete")
    print("Best accuracy = {}%".format(best_acc * 100))
    model.load_state_dict(best_model_state_dict)
    torch.save(model.state_dict(),
               'C:\\Users\QinJingChang\PycharmProjects\zkcup\Save model\\' + 'loss=' + str(min_loss)[:5] + '.tar')
    return model

# 只分类了30个键
def KeytypeToTarget(Keytype):
    total_class_n = 30
    Target = [0] * total_class_n
    MappingDict = {';':26, ',':27, '.':28, '/':29}
    if 'A' <= Keytype <= 'Z':
        index = int(Keytype) - 65
        Target[index] = 1
    elif Keytype in MappingDict:
        Target[MappingDict[Keytype]] = 1
    return Target

if __name__ == '__main__':
    sizelist = [1000, 300, 300, 30]
    model = network.FFTnet(sizelist)

    train_excel_dir = r'/User'                    # 换成excel文件的目录
    test_excel_dir = r'/User'
    dataset = {'train': FFTDataset(train_excel_dir),
               'test': FFTDataset(test_excel_dir)}
    dataloader = {x: DataLoader(dataset[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train', 'test']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    CostFunction = nn.CrossEntropyLoss()

    training_model(model)