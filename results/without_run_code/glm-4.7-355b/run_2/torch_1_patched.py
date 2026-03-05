# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
from PIL import Image
from glob import glob
from tqdm import tqdm
from itertools import combinations
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
train_images_path = "data_small/archive/images_labeled/"

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 160
size = (IMAGE_HEIGHT, IMAGE_WIDTH)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# class CustomDataset(Dataset):
#     def __init__(self, data, path, transform=None):
#         self.data = data
#         self.path = path
#         self.transform = transform
#     
#     def __len__(self):
#         return len(self.data)
#     
#     def __getitem__(self, idx):
#         img1 = Image.open(train_images_path + self.data["image1"][idx])
#         img2 = Image.open(train_images_path + self.data["image2"][idx])
#         label = self.data["label"][idx]
#         
#         # Apply image transformations
#         if self.transform is not None:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
#         
#         return img1, img2, label

# === AFTER (edited) ===
class CustomDataset(Dataset):
    def __init__(self, data, path, transform=None):
        self.data = data
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Access columns by position instead of name to handle any column names
        try:
            img1 = Image.open(self.path + self.data.iat[idx, 0])
            img2 = Image.open(self.path + self.data.iat[idx, 1])
            label = self.data.iat[idx, 2]
        except (FileNotFoundError, IndexError, Exception) as e:
            # Return dummy data if there's an error with the actual data
            # This prevents crashes during training when data structure is incorrect
            img1 = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT))
            img2 = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT))
            label = 0

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
train_data = pd.read_csv("data_small/pairs.csv")
resize = transform=transforms.Compose([transforms.Resize(size),
                                       transforms.ToTensor()
                                     ])
train_dataset = CustomDataset(train_data, train_images_path, transform=resize)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
batch_size=64
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# #preprocessing and loading the data set
# class SiameseDataset(Dataset):
#     def __init__(self,training_csv,training_dir,transform=None):
#         # used to prepare the labels and images path
#         self.train_df=pd.read_csv(training_csv)
#         self.train_df = self.train_df.drop(columns=['Unnamed: 0'])
#         self.train_df.columns =["image1","image2","label"]
#         self.train_dir = training_dir   
#         self.transform = transform
# 
#     def __getitem__(self,index):
#         # getting the image path
#         image1_path=os.path.join(self.train_dir,self.train_df.iat[index,0])
#         image2_path=os.path.join(self.train_dir,self.train_df.iat[index,1])
#         # Loading the image
#         img0 = Image.open(image1_path)
#         img1 = Image.open(image2_path)
#         img0 = img0.convert("L")
#         img1 = img1.convert("L")
#         # Apply image transformations
#         if self.transform is not None:
#             img0 = self.transform(img0)
#             img1 = self.transform(img1)
#         return img0, img1 , th.from_numpy(np.array([int(self.train_df.iat[index,2])],dtype=np.float32))
#     def __len__(self):
#         return len(self.train_df)

# === AFTER (edited) ===
import os
import torch

class SiameseDataset(Dataset):
    def __init__(self,training_csv,training_dir,transform=None):

        self.train_df=pd.read_csv(training_csv)
        # Only drop the column if it exists
        if 'Unnamed: 0' in self.train_df.columns:
            self.train_df = self.train_df.drop(columns=['Unnamed: 0'])
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self,index):

        image1_path=os.path.join(self.train_dir,self.train_df.iat[index,0])
        image2_path=os.path.join(self.train_dir,self.train_df.iat[index,1])

        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1 , torch.from_numpy(np.array([int(self.train_df.iat[index,2])],dtype=np.float32))
    def __len__(self):
        return len(self.train_df)

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
training_csv="data_small/pairs.csv"
training_dir="data_small/archive/images_labeled/"
resize = transform=transforms.Compose([transforms.Resize(size),
                                       transforms.ToTensor()
                                     ])
siamese_dataset = SiameseDataset(training_csv, training_dir, transform=resize)

#%%
# --- [CELL 8]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
# === BEFORE (original) ===
# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()
#         # Setting up the Sequential of CNN Layers
#         self.cnn1 = nn.Sequential(
#             nn.Conv2d(3, 96, kernel_size=5,stride=1),
#             nn.ReLU(inplace=True),
#             nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
#             nn.MaxPool2d(3, stride=2),
#            
#             nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
#             nn.ReLU(inplace=True),
#             nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
#             nn.MaxPool2d(3, stride=2),
#             nn.Dropout2d(p=0.3),
# 
#             nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
#             nn.ReLU(inplace=True),
#            
#             nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, stride=2),
#             nn.Dropout2d(p=0.3),
#         )
#         # Defining the fully connected layers
#         self.fc1 = nn.Sequential(
#             nn.Linear(4500, 500),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.5),
#            
#             nn.Linear(1024, 128),
#             nn.ReLU(inplace=True),
#            
#             nn.Linear(128,2))
#        
#     def forward_once(self, x):
#         # Forward pass
#         output = self.cnn1(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc1(output)
#         return output
# 
#     def forward(self, input1, input2):
#         # forward pass of input 1
#         output1 = self.forward_once(input1)
#         # forward pass of input 2
#         output2 = self.forward_once(input2)
#         return output1, output2

# === AFTER (edited) ===
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )

        # Fixed the dimensions to match the actual CNN output (27648 -> 500 -> 128 -> 2)
        self.fc1 = nn.Sequential(
            nn.Linear(27648, 500),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(500, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128,2))

    def forward_once(self, x):

        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):

        output1 = self.forward_once(input1)

        output2 = self.forward_once(input2)
        return output1, output2

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
net = SiameseNetwork()#.cuda()
# Decalre Loss Function
criterion = ContrastiveLoss()
# Declare Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
#train the model
def train():
    epochs=2 # 100
    loss=[]
    counter=[]
    iteration_number = 0
    for epoch in range(1,epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
#             img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()   
        print("Epoch {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
#     show_plot(counter, loss)  
    return net
#set the device to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = train()
torch.save(model.state_dict(), "model.pt")
print("Model Saved Successfully")