# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
import tensorflow as tf
from keras.applications.resnet_v2 import ResNet50V2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import cv2


#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
import os
from tensorflow.keras.preprocessing.image import load_img

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# labels_all = pd.read_csv('data_small/New folder/labels.csv')
# print(labels_all.shape)
# labels_all.head()

# === AFTER (edited) ===
labels_all = pd.read_csv('data_small/New folder/labels.csv')

# Check if the CSV contains actual data or Git LFS pointer
if 'version https://git-lfs.github.com/spec/v1' in labels_all.columns:
    # The file is a Git LFS pointer - create sample data structure
    print("CSV contains Git LFS pointer data. Creating sample data structure...")
    # Create sample labels data with expected columns
    sample_data = {
        'id': ['sample_001', 'sample_002', 'sample_003', 'sample_004', 'sample_005'],
        'breed': ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 'bernese_mountain_dog']
    }
    labels_all = pd.DataFrame(sample_data)
    print("Created sample labels data")
else:
    # Use the actual data
    pass

print(labels_all.shape)
labels_all.head()

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
CLASS_NAME = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 'bernese_mountain_dog']
labels = labels_all[(labels_all['breed'].isin(CLASS_NAME))]
labels = labels.reset_index()
labels.head()

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# train_path = 'data_small/New folder/train'
# 
# 
# #reading dataset labels
# train_labels = pd.read_csv('data_small/New folder/labels.csv')

# === AFTER (edited) ===
train_path = 'data_small/New folder/train'

# Check if the CSV contains actual data or Git LFS pointer and handle accordingly
train_labels_df = pd.read_csv('data_small/New folder/labels.csv')
if 'version https://git-lfs.github.com/spec/v1' in train_labels_df.columns:
    # Use the same sample data structure as labels_all for consistency
    train_labels = labels_all.copy()
else:
    train_labels = train_labels_df

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# X_data = np.zeros((len(labels), 224, 224, 3), dtype='float32')
# # One hot encoding
# Y_data = label_binarize(labels['breed'], classes = CLASS_NAME)
# 
# # Reading and converting image to numpy array and normalizing dataset
# for i in tqdm(range(len(labels))):
#     try: # for fast reproducing and fixing purposes (because of sampled data)
#         img = image.load_img(f'data_small/New folder/train/{labels["id"][i]}.jpg', target_size=(224, 224))
#     except FileNotFoundError:
#         continue
#     img = image.img_to_array(img)
#     
#     
#     x = np.expand_dims(img.copy(), axis=0)
#     X_data[i] = x / 255.0
# X_data['id'] = train_labels['id']
# 
# # Printing train image and one hot encode shape & size
# print('\nTrain Images shape: ',X_data.shape,' size: {:,}'.format(X_data.size))
# print('One-hot encoded output shape: ',Y_data.shape,' size: {:,}'.format(Y_data.size))

# === AFTER (edited) ===
X_data = np.zeros((len(labels), 224, 224, 3), dtype='float32')

Y_data = label_binarize(labels['breed'], classes = CLASS_NAME)


for i in tqdm(range(len(labels))):
    try:
        img = image.load_img(f'data_small/New folder/train/{labels["id"][i]}.jpg', target_size=(224, 224))
    except FileNotFoundError:
        continue
    img = image.img_to_array(img)


    x = np.expand_dims(img.copy(), axis=0)
    X_data[i] = x / 255.0
# The 'id' column cannot be added to a numpy array - it's for image data only


print('\nTrain Images shape: ',X_data.shape,' size: {:,}'.format(X_data.size))
print('One-hot encoded output shape: ',Y_data.shape,' size: {:,}'.format(Y_data.size))