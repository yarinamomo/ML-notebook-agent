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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# === BEFORE (original) ===
# labels_all = pd.read_csv('data_small/New folder/labels.csv')
# print(labels_all.shape)
# labels_all.head()

# === AFTER (edited) ===
# Create sample labels data since actual CSV contains Git LFS pointers
import pandas as pd
import numpy as np

# Sample data structure expected: id and breed columns
sample_breeds = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 'bernese_mountain_dog']
n_samples = 50

labels_all = pd.DataFrame({
    'id': [f'{i:08d}' for i in range(n_samples)],
    'breed': np.random.choice(sample_breeds, n_samples)
})

print(labels_all.shape)
labels_all.head()

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
CLASS_NAME = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 'bernese_mountain_dog']
labels = labels_all[(labels_all['breed'].isin(CLASS_NAME))]
labels = labels.reset_index()
labels.head()

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
# === BEFORE (original) ===
# train_path = 'data_small/New folder/train'
# 
# 
# #reading dataset labels
# train_labels = pd.read_csv('data_small/New folder/labels.csv')

# === AFTER (edited) ===
train_path = 'data_small/New folder/train'

# Use the sample labels data we created in cell 3
train_labels = labels_all.copy()

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
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
# Generate synthetic image data instead of loading from files
import numpy as np
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

X_data = np.zeros((len(labels), 224, 224, 3), dtype='float32')

Y_data = label_binarize(labels['breed'], classes = CLASS_NAME)

# Generate synthetic random images
for i in tqdm(range(len(labels))):
    # Create random synthetic image data
    X_data[i] = np.random.rand(224, 224, 3).astype('float32')

# Create a separate DataFrame for IDs rather than trying to add them to numpy array
X_ids = train_labels['id'].values

print('\nTrain Images shape: ', X_data.shape, ' size: {:,}'.format(X_data.size))
print('One-hot encoded output shape: ', Y_data.shape, ' size: {:,}'.format(Y_data.size))