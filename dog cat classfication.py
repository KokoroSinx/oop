# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''
def between_min_max(distincts):
    maxium = max(distincts)
    minium = min(distincts)
    if distincts.index(maxium) < distincts.index(minium):
        return distincts[distincts.index(maxium):distincts.index(minium)+1]
    else:
        return distincts[distincts.index(minium):distincts.index(maxium)+1]
    
assert between_min_max([5,3,1,4,6,2,0]) == [6,2,0]
assert between_min_max([5,3,1,4,6,2]) == [1,4,6]
assert between_min_max([5,3,1,4,2]) == [5,3,1]
assert between_min_max([1]) == [1]
'''


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# 超参数
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# 数据转换
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 训练集和测试集
train_dataset = torchvision.datasets.ImageFolder(root='train/',
                                                 transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(root='test/',
                                                transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

classes = ('cat', 'dog')

# 定义模型
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # 每个epoch之后评估模型
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the classfication is {correct}")
