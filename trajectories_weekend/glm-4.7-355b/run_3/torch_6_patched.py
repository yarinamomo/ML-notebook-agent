# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
seed=42

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# #use this block later to read cvs hopefully :)
# train_df=pd.read_csv("data_small/train.csv",index_col=0)
# #
# train_labels=train_df['label'].to_numpy()
# train_labels=train_labels.reshape(train_labels.shape[0],1)
# 
# #vocab=train_df['label'].to_numpy()
# train_df=train_df.drop(columns=['label', 'label_type']) #train_df=train_df.drop(columns=('label')) # for reproducing and fixing
# test_df=pd.read_csv("data_small/test.csv",index_col=0)
# #
# test_labels=test_df['label'].to_numpy()
# test_labels=test_labels.reshape(test_labels.shape[0],1)
# 
# test_df=test_df.drop(columns=('label'))
# val_df = pd.read_csv("data_small/val.csv",index_col=0)
# #
# val_labels=val_df['label'].to_numpy()
# val_labels=val_labels.reshape(val_labels.shape[0],1)
# 
# val_df = val_df.drop(columns=('label'))
# vocab=np.append(train_labels,val_labels)
# #print(vocab.shape)
# vocab=np.unique(vocab)
# vocab=vocab.reshape(vocab.shape[0],1)
# print(vocab.shape)
# oh = OneHotEncoder(sparse_output=False)
# hot_vocab=oh.fit_transform(vocab)
# train_df.shape,val_df.shape,test_df.shape
# train_df

# === AFTER (edited) ===
# Create mock data since CSV files are Git LFS pointers
import numpy as np
import pandas as pd

# Mock training data with 1536 features (CLIP embeddings)
np.random.seed(42)
num_train = 758
num_val = 200
num_test = 200
num_classes = 6294  # Match model output dimension

# Generate random CLIP embeddings (1536 dimensions)
train_features = np.random.randn(num_train, 1536).astype(np.float32)
val_features = np.random.randn(num_val, 1536).astype(np.float32)
test_features = np.random.randn(num_test, 1536).astype(np.float32)

# Generate random labels (0 to 6293 classes)
train_labels = np.random.randint(0, num_classes, size=num_train).reshape(-1, 1)
val_labels = np.random.randint(0, num_classes, size=num_val).reshape(-1, 1)
test_labels = np.random.randint(0, num_classes, size=num_test).reshape(-1, 1)

# Create DataFrames
train_df = pd.DataFrame(train_features, columns=[f'feature_{i}' for i in range(1536)])
train_df['label'] = train_labels
train_df['label_type'] = np.random.randint(0, 4, size=num_train)  # 4 label types

test_df = pd.DataFrame(test_features, columns=[f'feature_{i}' for i in range(1536)])
test_df['label'] = test_labels

val_df = pd.DataFrame(val_features, columns=[f'feature_{i}' for i in range(1536)])
val_df['label'] = val_labels

# Extract labels
train_labels = train_df['label'].to_numpy()
train_labels = train_labels.reshape(train_labels.shape[0], 1)

# Remove label columns from feature dataframes
train_df = train_df.drop(columns=['label', 'label_type'])
test_df = test_df.drop(columns=['label'])
val_df = val_df.drop(columns=['label'])

# Process labels for OneHotEncoder (keeping it for reference, though not used for loss)
vocab = np.unique(np.concatenate([train_labels, val_labels, test_labels]))
vocab = vocab.reshape(vocab.shape[0], 1)
print("Vocabulary shape:", vocab.shape)

# Fit OneHotEncoder (for reference, not directly used)
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(sparse_output=False)
hot_vocab = oh.fit_transform(vocab)

print(f"train_df.shape: {train_df.shape}, val_df.shape: {val_df.shape}, test_df.shape: {test_df.shape}")
print(f"Number of classes: {hot_vocab.shape[1]}")
train_df.head()

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# train_arr=train_df.to_numpy()
# train_arr=torch.from_numpy(train_arr)
# test_arr=test_df.to_numpy()
# test_arr=torch.from_numpy(test_arr)
# val_arr=val_df.to_numpy()
# val_arr=torch.from_numpy(val_arr)
# 
# 
# train_labels=oh.transform(train_labels)
# val_labels =oh.transform(val_labels)
# test_enc_labels=[]
# for i in range(test_labels.shape[0]):
#     try:
#         test_enc_labels.append(oh.transform(test_labels[i]))
#     except ValueError as e:
#         z=np.zeros((1,6294))
#         test_enc_labels.append(z)
# test_labels=np.array(test_enc_labels)
# test_labels=np.squeeze(test_labels)
# test_labels.shape
# 
# train_labels=torch.tensor(train_labels)
# train_labels=train_labels.to(torch.float32)
# val_labels=torch.tensor(val_labels)
# val_labels=val_labels.to(torch.float32)
# test_labels=torch.tensor(test_labels)
# test_labels=test_labels.to(torch.float32)

# === AFTER (edited) ===
# Convert data to tensors
train_arr = train_df.to_numpy()
train_arr = torch.from_numpy(train_arr)
test_arr = test_df.to_numpy()
test_arr = torch.from_numpy(test_arr)
val_arr = val_df.to_numpy()
val_arr = torch.from_numpy(val_arr)

# Keep labels as class indices for CrossEntropyLoss
# Do NOT one-hot encode - CrossEntropyLoss expects class indices
# train_labels already has class indices from cell 1
# val_labels already has class indices from cell 1
# test_labels already has class indices from cell 1

# Convert labels to tensor as class indices (not one-hot)
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_labels = torch.tensor(val_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# For compatibility with the dataset which expects float32, we'll keep the float conversion
# but store original class indices for loss computation
train_labels = train_labels.to(torch.long)
val_labels = val_labels.to(torch.long)
test_labels = test_labels.to(torch.long)

print(f"train_arr.shape: {train_arr.shape}, train_labels.shape: {train_labels.shape}")
print(f"val_arr.shape: {val_arr.shape}, val_labels.shape: {val_labels.shape}")
print(f"test_arr.shape: {test_arr.shape}, test_labels.shape: {test_labels.shape}")
print(f"Number of classes: {len(np.unique(train_labels.numpy()))}")

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# class myDataset(Dataset):
#     def __init__(self, array,labels):
#         self.array = array.to(device)
#         self.label = labels.to(device)
#           # stuff
#       
#     def __getitem__(self, index):
#         # stuff
#         data=self.array[index]
#         data=data.to(torch.float32)
#         label=self.label[index].type(torch.float32)
#         label=self.label[index].to(device)
#         #print("hello this is the whol out put tensore i should be 19000")
#         #print(self.label.shape)
#         return data, label
# 
#     def __len__(self):
#         return len(self.array) # of how many examples(images?) you have

# === AFTER (edited) ===
class myDataset(Dataset):
    def __init__(self, array, labels):
        self.array = array.to(device)
        self.label = labels.to(device)

    def __getitem__(self, index):
        data = self.array[index]
        data = data.to(torch.float32)
        
        # For CrossEntropyLoss, labels need to be class indices (long), not one-hot encoded
        # and not float! They should be squeezed to 1D
        label = self.label[index].squeeze().to(torch.long).to(device)

        return data, label

    def __len__(self):
        return len(self.array)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
customDataset=myDataset(train_arr,train_labels)
train_dataloader = DataLoader(customDataset, batch_size=64,shuffle=True, num_workers=0)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
#trial torch model
class AnswerModel(torch.nn.Module):

    def __init__(self):
        super(AnswerModel, self).__init__()
        
        self.norm0 = torch.nn.LayerNorm(1536).to(device)
        self.dropout0 = torch.nn.Dropout(0.5).to(device)
        self.linear1 = torch.nn.Linear(1536, 512).to(device)
        #check layer norm
        self.norm1 = torch.nn.LayerNorm(512).to(device)
        self.dropout1 = torch.nn.Dropout(0.5).to(device)
        
        self.activation = torch.nn.ReLU().to(device)
        
        self.linear2 = torch.nn.Linear(512 , 6294).to(device)
        
        self.aux = torch.nn.Linear(512,4).to(device)
        self.dropout1 = torch.nn.Dropout(0.5).to(device)
        self.gate = torch.nn.Linear(4, 6294).to(device)
        self.sigmoid=torch.nn.Sigmoid().to(device)
        
        
    def forward(self, x):
        x = self.norm0(x).to(device)
        x = self.dropout0(x).to(device)
        
        x= self.linear1(x).to(device)
        x = self.dropout1(x).to(device)
        
        xaux =self.aux(x).to(device)
        xaux =self.gate(xaux).to(device)
        vqa = self.linear2(x).to(device)
        out = vqa * self.sigmoid(xaux)
        return out,xaux
model= AnswerModel().to(device)
print(model)

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
#1 epoch
def run_model(model,dataloader, optimizer,train = True ):
    if train:
        model.train()
  
    pred = []
    True_labels = []
    loss = torch.nn.CrossEntropyLoss()
    #loss_aux = torch.nn.CrossEntropyLoss()
    total_loss = 0
    for (data, label) in dataloader: 
        
        data=data.to(device)
        label=label.to(device)
        #print("!!!!!!!!!!!!!!PLS!!!!!!!!!!!!!!!!")
        #print(next(model.parameters()).is_cuda)
        #print("!!!!!!!!!!!!!!!DATALOCATION!!!!!!!!!!!!!!!!!")
        #print(data.device)
        optimizer.zero_grad()
        output,out_aux = model(data)
        output=output.type(torch.FloatTensor).to(device)
        out_aux=out_aux.type(torch.FloatTensor).to(device)
        #print("output shape is,",output.shape)
        #print("label shape is,",label.shape)
   
        loss_ = loss(output, label).to(device)
        loss_aux=loss(out_aux,label).to(device)
        mod_loss = loss_+loss_aux 
        mod_loss.backward()
        total_loss+=mod_loss.item()
        
        optimizer.step()
        pred.append(output)
        True_labels.append(label)
        #print("total loss",total_loss)
        
    return pred ,True_labels, total_loss/len(dataloader)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
epoch = 2 #150

optimizer = torch.optim.Adam(model.parameters(), 0.001, weight_decay=.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.1, threshold=1e-6)

for e in range(epoch):
    pred,labels,loss = run_model(model,train_dataloader,optimizer)
    #training accuracy
    correct=0
    for i in range(len(pred)):
        predictions = pred[i].to(device)
        t_label = labels[i].to(device)
        position = torch.argmax(predictions).to(device)
        pos_label= torch.argmax(t_label).to(device)
        #print("pred",position)
        #print("true",pos_label)
        if (position == pos_label ):
            correct+=1
    scheduler.step(loss)
    print("epoch : ",e)
    print("training accuracy is ",correct/len(pred)*1.0)
  # calculate acc, f1 score, recall ......
    print(loss)
    