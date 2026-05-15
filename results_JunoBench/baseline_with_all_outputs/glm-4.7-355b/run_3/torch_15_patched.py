# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
jsdf = pd.read_json('data/train_annotations')
jsdf.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
Id = []

import os
for dirname, _, filenames in os.walk('data/train/train'):
    for filename in filenames:
        Id.append(os.path.join(dirname, filename))
Id[:5]

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
train = pd.DataFrame()
train = train.assign(filename = Id)
train['image_id'] = train['filename'].str.replace('data/train/train/image_id_','')
train['image_id'] = train['image_id'].str.replace('.jpg','')
train['image_id'] = train['image_id'].astype(int)
train.head()

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
train_data = pd.merge(train,jsdf,on='image_id',how='outer')
train_data = train_data[['filename','category_id']]
train_data.columns = ['filename','label']
train_data.head()

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
train_data['filename'] = train_data['filename'].str.replace('data/train/train/','')

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
train_data['label'] = train_data['label'].replace({1:0,2:1})

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
transform = transforms.Compose([
    transforms.CenterCrop((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
data_path = 'data/train/train'
images = []
targets = []

for i,annotation in train_data.iterrows():
    image_name = annotation['filename']
    target = annotation['label']
    image_path = os.path.join(data_path, image_name)
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    images.append(image)
    targets.append(torch.tensor(target))

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
image_tensor = torch.stack(images)
target_tensor = torch.stack(targets)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
dataset = torch.utils.data.TensorDataset(image_tensor, target_tensor)

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
from torch.utils.data import random_split
train_dataset, test_dataset = random_split(dataset, [400, 100])

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 128 * 128, 128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = x.view(-1, 32 * 128 * 128)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
    
model = CNN()#.to('cuda')
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
def test_model(model):
    model.eval()
    correct = 0
    total = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test data: {accuracy:.2f}%')

test_model(model)

#%%
# --- [CELL 15]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 16}
# === BEFORE (original) ===
# n_epoch = 1
# for epoch in range(n_epoch):
#     model.train()  
# 
#     for batch_idx, (sekil, netice) in enumerate(train_loader):
#         #sekil, netice = sekil.to('cuda'), netice.to('cuda')
#         optimizer.zero_grad()
# 
#         outputs = model(sekil)
#         loss = loss_fn(outputs, netice)
# 
#         loss.backward()
#         optimizer.step()
# 
#         if (batch_idx + 1) % 4 == 0:
#             print(f"Epoch [{epoch+1}/{n_epoch}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}")

# === AFTER (edited) ===
n_epoch = 1
for epoch in range(n_epoch):
    model.train()

    for batch_idx, (sekil, netice) in enumerate(train_loader):

        optimizer.zero_grad()

        outputs = model(sekil)
        # Reshape netice to match outputs shape for BCELoss
        netice = netice.unsqueeze(1)
        loss = loss_fn(outputs, netice)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 4 == 0:
            print(f"Epoch [{epoch+1}/{n_epoch}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}")