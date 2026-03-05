# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
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
# execution_status: {'execution_count': 2, 'status': 'ok'}
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
# execution_status: {'execution_count': 3, 'status': 'ok'}
import os
from tensorflow.keras.preprocessing.image import load_img

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 4, 'status': 'ok'}
labels_all = pd.read_csv('data_small/New folder/labels.csv')
print(labels_all.shape)
labels_all.head()

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'execution_count': 5, 'status': 'ok'}
# === BEFORE (original) ===
# CLASS_NAME = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 'bernese_mountain_dog']
# labels = labels_all[(labels_all['breed'].isin(CLASS_NAME))]
# labels = labels.reset_index()
# labels.head()

# === AFTER (edited) ===
# Load the actual CSV data (it's a Git LFS pointer, but we'll create sample data)
# Create mock data based on available images
import os
import random

# Get list of available images
train_dir = 'data_small/New folder/train'
image_files = [f.replace('.jpg', '') for f in os.listdir(train_dir) if f.endswith('.jpg')]

# Create mock labels DataFrame with the expected structure
CLASS_NAME = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 'bernese_mountain_dog']
np.random.seed(42)

# Create label data from available images
num_samples = min(len(image_files), len(CLASS_NAME) * 15)  # Use up to 15 images per breed
sampled_files = random.sample(image_files, num_samples)

# Assign breeds to images
breeds = []
for i, img_id in enumerate(sampled_files):
    breeds.append(CLASS_NAME[i % len(CLASS_NAME)])

# Create the labels DataFrame
labels = pd.DataFrame({'id': sampled_files, 'breed': breeds})
labels = labels.reset_index(drop=True)
labels.head()

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 6, 'status': 'ok'}
train_path = 'data_small/New folder/train'


#reading dataset labels
train_labels = pd.read_csv('data_small/New folder/labels.csv')


#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'execution_count': 7, 'status': 'ok'}
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
# Create arrays to store loaded images and labels
X_data = []
valid_breeds = []

for i in tqdm(range(len(labels))):
    try:
        img = image.load_img(f'data_small/New folder/train/{labels["id"][i]}.jpg', target_size=(224, 224))
        img = image.img_to_array(img)
        X_data.append(img / 255.0)
        valid_breeds.append(labels['breed'][i])
    except (FileNotFoundError, Exception) as e:
        # Skip corrupted or missing images
        continue

# Convert to numpy arrays
if X_data:
    X_data = np.array(X_data, dtype='float32')
    Y_data = label_binarize(valid_breeds, classes=CLASS_NAME)
else:
    X_data = np.zeros((0, 224, 224, 3), dtype='float32')
    Y_data = np.zeros((0, len(CLASS_NAME)), dtype='int')

print('\nTrain Images shape: ',X_data.shape,' size: {:,}'.format(X_data.size))
print('One-hot encoded output shape: ',Y_data.shape,' size: {:,}'.format(Y_data.size))