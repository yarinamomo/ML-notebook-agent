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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
labels_all = pd.read_csv('data_small/New folder/labels.csv')
print(labels_all.shape)
labels_all.head()

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# CLASS_NAME = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 'bernese_mountain_dog']
# labels = labels_all[(labels_all['breed'].isin(CLASS_NAME))]
# labels = labels.reset_index()
# labels.head()

# === AFTER (edited) ===
CLASS_NAME = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 'bernese_mountain_dog']

# Check if 'breed' column exists
if 'breed' not in labels_all.columns:
    print("Error: 'breed' column not found in labels_all")
    print(f"Available columns: {labels_all.columns.tolist()}")
    print("\nThe CSV file may need to be pulled from Git LFS.")
    print("Please run: git lfs pull")
    labels = pd.DataFrame()
else:
    labels = labels_all[(labels_all['breed'].isin(CLASS_NAME))]
    labels = labels.reset_index()

labels.head()

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
train_path = 'data_small/New folder/train'


#reading dataset labels
train_labels = pd.read_csv('data_small/New folder/labels.csv')


#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 7}
X_data = np.zeros((len(labels), 224, 224, 3), dtype='float32')
# One hot encoding
Y_data = label_binarize(labels['breed'], classes = CLASS_NAME)

# Reading and converting image to numpy array and normalizing dataset
for i in tqdm(range(len(labels))):
    try: # for fast reproducing and fixing purposes (because of sampled data)
        img = image.load_img(f'data_small/New folder/train/{labels["id"][i]}.jpg', target_size=(224, 224))
    except FileNotFoundError:
        continue
    img = image.img_to_array(img)
    
    
    x = np.expand_dims(img.copy(), axis=0)
    X_data[i] = x / 255.0
X_data['id'] = train_labels['id']

# Printing train image and one hot encode shape & size
print('\nTrain Images shape: ',X_data.shape,' size: {:,}'.format(X_data.size))
print('One-hot encoded output shape: ',Y_data.shape,' size: {:,}'.format(Y_data.size))