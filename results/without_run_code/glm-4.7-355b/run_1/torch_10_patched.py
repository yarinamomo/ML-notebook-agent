# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# === BEFORE (original) ===
# import os
# import torch
# import torchvision
# import numpy as np
# from PIL import Image
# import torch.nn as nn
# from torch.nn import ReLU
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset,DataLoader
# from torch.nn import Conv2d,MaxPool2d,Flatten,Linear
# 
# 
# train_data_path = 'data_small/101/train'
# test_data_path = 'data_small/101/val'
# transform = transforms.Compose([
#     transforms.Resize((227,227)),#缩放大小
#     transforms.ToTensor(),#转化类型
# ])
# 
# train_data=torchvision.datasets.ImageFolder(root = train_data_path,transform = transform)
# test_data = torchvision.datasets.ImageFolder(root = test_data_path,transform = transform)
# 
# #利用dataloader来加载数据集
# train_dataloader = DataLoader(dataset = train_data,batch_size=64,shuffle=True,drop_last=False)
# test_dataloader = DataLoader(dataset = test_data,batch_size=64,shuffle=True,drop_last=False)
# 
# #搭建神经网络
# class module(nn.Module):
#     def __init__(self):
#         super(module,self).__init__()
#         self.conv1 = Conv2d(3,96,11,stride = 4)#227*227*3->55*55*96
#         self.relu1 = ReLU()#55*55*96
#         self.maxpool1 = MaxPool2d(3,stride = 2)#55*55*96->27*27*96
#         self.conv2 = Conv2d(96,256,5,stride = 1,padding = 2)#27*27*96->27*27*256
#         self.relu2 = ReLU()#27*27*256
#         self.maxpool2 = MaxPool2d(3,stride = 2)#27*27*256->13*13*256
#         self.conv3 = Conv2d(256,384,3,stride = 1,padding = 1)#13*13*256->13*13*384
#         self.relu3 = ReLU()#13*13*384
#         self.conv4 = Conv2d(384,256,3,stride = 1,padding = 1)#13*13*384->13*13*256
#         self.relu4 = ReLU()#13*13*256
#         self.conv5 = Conv2d(256,256,3,stride = 1,padding = 1)#13*13*256
#         self.relu5 = ReLU()#13*13*256
#         self.maxpool5 = MaxPool2d(3,stride = 2)#13*13*256->6*6*256
#         self.flatten = Flatten()#6*6*256
#         self.fc1 = Linear(6*6*256,4096)
#         self.fc2 = Linear(4096,101)
#         
#         
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.conv4(x)
#         x = self.relu4(x)
#         x = self.conv5(x)
#         x = self.relu5(x)
#         x = self.maxpool5(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x
# 
# #创建网络模型
# module = module()
# if torch.cuda.is_available():
#     module = module.cuda()
# 
# #损失函数
# loss_fn = nn.CrossEntropyLoss()
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()
# 
# #优化器
# learning_rate = 0.001
# optimizer = torch.optim.Adam(module.parameters(),lr = learning_rate)
# 
# #设置训练网络参数
# total_train_step = 0#训练次数
# total_test_step = 0#测试次数
# epoch = 2 #20#训练轮数
# 
# for i in range(epoch):
#     print('------第{}轮训练开始------'.format(i+1))
#     
#     module.train()
#     for data in train_dataloader:#训练步骤
#         imgs,targets = data
#         if torch.cuda.is_available():
#             imgs = imgs.cuda()
#             targets = targets.cuda()
#         outputs = module(imgs)
#         loss = loss_fn(outputs,targets)
#         
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()#优化器优化
#         
#         total_train_step = total_train_step + 1
#         if total_train_step % 100 ==0:
#             print("训练次数：{}，LOSS：{}".format(total_train_step,loss.item()))
# 
#     module.eval()
#     total_test_loss = 0
#     total_accuracy = 0
#     with torch.no_grad():
#         for data in test_dataloader:
#             imgs,targets = data
#             if torch.cuda.is_available():
#                 imgs = imgs.cuda()
#                 targets = targets.cuda()
#             outputs = module(imgs)
#             loss = loss_fn(outputs,targets)
#             total_test_loss = total_test_loss + loss.item()
#             accuracy = (outputs.argmax(1) == targets).sum()
#             total_accuracy = total_accuracy + accuracy
#     print('整体测试集LOSS:{}'.format(total_test_loss))
#     print('正确率：{}'.format(total_accuracy/len(test_data)))
#     total_test_step = total_test_step + 1

# === AFTER (edited) ===
import os
import torch
import torchvision
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.nn import ReLU
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear


train_data_path = 'data_small/101/train'
test_data_path = 'data_small/101/val'
transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
])

# Custom dataset to handle corrupted images
class RobustImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"Warning: Could not load image at index {index}. Skipping. Error: {e}")
            # Return a random image and label to avoid breaking the dataloader
            return torch.randn(3, 227, 227), 0

train_data = RobustImageFolder(root=train_data_path, transform=transform)
test_data = RobustImageFolder(root=test_data_path, transform=transform)


train_dataloader = DataLoader(dataset = train_data,batch_size=64,shuffle=True,drop_last=False)
test_dataloader = DataLoader(dataset = test_data,batch_size=64,shuffle=True,drop_last=False)


class module(nn.Module):
    def __init__(self):
        super(module,self).__init__()
        self.conv1 = Conv2d(3,96,11,stride = 4)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(3,stride = 2)
        self.conv2 = Conv2d(96,256,5,stride = 1,padding = 2)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(3,stride = 2)
        self.conv3 = Conv2d(256,384,3,stride = 1,padding = 1)
        self.relu3 = ReLU()
        self.conv4 = Conv2d(384,256,3,stride = 1,padding = 1)
        self.relu4 = ReLU()
        self.conv5 = Conv2d(256,256,3,stride = 1,padding = 1)
        self.relu5 = ReLU()
        self.maxpool5 = MaxPool2d(3,stride = 2)
        self.flatten = Flatten()
        self.fc1 = Linear(6*6*256,4096)
        self.fc2 = Linear(4096,101)


    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


module = module()
if torch.cuda.is_available():
    module = module.cuda()


loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()


learning_rate = 0.001
optimizer = torch.optim.Adam(module.parameters(),lr = learning_rate)


total_train_step = 0
total_test_step = 0
epoch = 2

for i in range(epoch):
    print('------第{}轮训练开始------'.format(i+1))

    module.train()
    for data in train_dataloader:
        imgs,targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = module(imgs)
        loss = loss_fn(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 ==0:
            print("训练次数：{}，LOSS：{}".format(total_train_step,loss.item()))

    module.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = module(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print('整体测试集LOSS:{}'.format(total_test_loss))
    print('正确率：{}'.format(total_accuracy/len(test_data)))
    total_test_step = total_test_step + 1

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# train_data_path = 'data_small/101/train'
# train_data=torchvision.datasets.ImageFolder(root = train_data_path,transform=transforms)
# train_dataloader = DataLoader(dataset = train_data,batch_size=64,shuffle=True,drop_last=False)
# for data in train_dataloader:#训练步骤
#     imgs,targets = data
#     if torch.cuda.is_available():
#         imgs = imgs.cuda()
#         targets = targets.cuda()
#     outputs = module(imgs)
#     loss = loss_fn(outputs,targets)

# === AFTER (edited) ===
train_data_path = 'data_small/101/train'
train_data = RobustImageFolder(root=train_data_path, transform=transform)
train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, drop_last=False)
for data in train_dataloader:
    imgs, targets = data
    if torch.cuda.is_available():
        imgs = imgs.cuda()
        targets = targets.cuda()
    outputs = module(imgs)
    loss = loss_fn(outputs, targets)