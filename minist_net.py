import pdb
import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

EPOCH = 30

# 训练集乱序，测试集有序
# Design model using class ------------------------------------------------------------------------------
class MinistNet(torch.nn.Module):
    def __init__(self):
        super(MinistNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 12)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）

# 反归一化函数
def denormalize(tensor, mean, std):
    mean = mean.view(1, 3, 1, 1)  # 调整均值形状为 (1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)    # 调整标准差形状为 (1, 3, 1, 1)
    return tensor * std + mean

def test(epoch, test_loader, model):
    correct = 0
    total = 0
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
            # tensor = denormalize(images, mean, std)
            # grid = torchvision.utils.make_grid(tensor, nrow=8, padding=2)
            # transform = torchvision.transforms.ToPILImage()
            # image = transform(grid)
            # image.save('debug/tmp.jpg')
            # pdb.set_trace()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc


# Train and Test CLASS --------------------------------------------------------------------------------------
# 把单独的一轮一环封装在函数类里
def train(epoch, train_loader, model, criterion, optimizer):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零

        # torch.save(model.state_dict(), './model_Mnist.pth')
        # torch.save(optimizer.state_dict(), './optimizer_Mnist.pth')


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.cls_dir = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14]
        self.transform = transform
        self.data_list = []
        for cdx, cls in enumerate(self.cls_dir):
            fl_list = os.listdir(f'{root_dir}/{cls}')
            if cls == 6:
                fl_list = fl_list * 10
            if cls == 5:
                fl_list = fl_list * 8
            for fl in fl_list:
                self.data_list.append((f'{root_dir}/{cls}/{fl}', cdx))

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        img = Image.open(self.data_list[idx][0])
        img = self.transform(img)
        return img, self.data_list[idx][1]



def main():
    # Super parameter ------------------------------------------------------------------------------------
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.5
    
    # Prepare dataset ------------------------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=3),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=(28, 28), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    # softmax归一化指数函数(https://blog.csdn.net/lz_peter/article/details/84574716),其中0.1307是mean均值和0.3081是std标准差

    train_dataset = CustomDataset(f'data/minesweeper', transform=transform)  # 本地没有就加上download=True
    test_dataset = CustomDataset(f'data/minesweeper', transform=transform)  # train=True训练集，=False测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = MinistNet()
    # Construct loss and optimizer ------------------------------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch, train_loader, model, criterion, optimizer)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test(epoch, test_loader, model)
        acc_list_test.append(acc_test)
        torch.save(model.state_dict(), f'ckpt/epoch{epoch:02d}.pth')

# Start train and Test --------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
    
