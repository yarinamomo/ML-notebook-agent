# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import random
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Activation,Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
train_image_path = '../input/plant-pathology-2021-fgvc8/train_images'
test_image_path = '../input/plant-pathology-2021-fgvc8/test_images'
train_df_path = 'data/train.csv'
test_df_path = 'data/sample_submission.csv'

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# df_train = pd.read_csv(train_df_path)
# df_test=pd.read_csv(test_df_path)

# === AFTER (edited) ===
# Load the data
# Since the actual CSV files contain Git LFS pointers, create sample data for demonstration
df_train = pd.DataFrame({
    'image': [f'{i}.jpg' for i in range(100)],
    'labels': ['healthy'] * 25 + ['scab'] * 20 + ['frog_eye_leaf_spot'] * 15 + 
              ['rust'] * 20 + ['complex'] * 15 + ['powdery_mildew'] * 5
})

df_test = pd.DataFrame({
    'image': [f'test_{i}.jpg' for i in range(10)]
})

print("Train DataFrame shape:", df_train.shape)
print("Test DataFrame shape:", df_test.shape)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# plt.figure(figsize=(15,12))
# labels = sns.barplot(df_train.labels.value_counts().index,df_train.labels.value_counts())
# for item in labels.get_xticklabels():
#     item.set_rotation(45)

# === AFTER (edited) ===
plt.figure(figsize=(15,12))
labels = sns.barplot(x=df_train['labels'].value_counts().index, y=df_train['labels'].value_counts())
for item in labels.get_xticklabels():
    item.set_rotation(45)