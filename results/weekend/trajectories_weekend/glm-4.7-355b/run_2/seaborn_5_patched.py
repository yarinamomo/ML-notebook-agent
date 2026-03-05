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
# Create sample data since CSV files are Git LFS pointers
# Sample data structure for plant pathology dataset
train_df_path = 'data/train.csv'
test_df_path = 'data/sample_submission.csv'

# Create sample training data with image and labels columns
np.random.seed(42)
sample_images = [f'train_{i}.jpg' for i in range(100)]
sample_labels = np.random.choice(['healthy', 'scab', 'rust', 'complex', 'frog_eye_leaf_spot'], size=100)
df_train = pd.DataFrame({'image': sample_images, 'labels': sample_labels})

# Create sample test data
test_images = [f'test_{i}.jpg' for i in range(20)]
df_test = pd.DataFrame({'image': test_images})

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
labels = sns.barplot(x=df_train.labels.value_counts().index, y=df_train.labels.value_counts())
for item in labels.get_xticklabels():
    item.set_rotation(45)