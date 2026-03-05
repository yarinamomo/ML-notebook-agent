# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
train_image_path = '../input/plant-pathology-2021-fgvc8/train_images'
test_image_path = '../input/plant-pathology-2021-fgvc8/test_images'
train_df_path = 'data/train.csv'
test_df_path = 'data/sample_submission.csv'

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'timeout', 'done': True, 'execution_count': None}
# === BEFORE (original) ===
# df_train = pd.read_csv(train_df_path)
# df_test=pd.read_csv(test_df_path)

# === AFTER (edited) ===
# Check what files are available
import os
print("Current directory:", os.getcwd())
print("\nFiles and directories:")
for item in os.listdir('.'):
    print(f"  {item}")

# Try to find data directory or CSV files
if os.path.exists('data'):
    print("\nContents of 'data' directory:")
    for item in os.listdir('data'):
        print(f"  {item}")

# Try loading the paths from cell 1
print(f"\nTrain CSV path: {train_df_path}")
print(f"Test CSV path: {test_df_path}")

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# plt.figure(figsize=(15,12))
# labels = sns.barplot(df_train.labels.value_counts().index,df_train.labels.value_counts())
# for item in labels.get_xticklabels():
#     item.set_rotation(45)

# === AFTER (edited) ===
# Print column names first to debug
print("DataFrame columns:", df_train.columns.tolist())
print("First few rows:")
print(df_train.head())

# Then try the plotting
plt.figure(figsize=(15,12))
# Get the first column that's not 'image' or use the first column if no match
label_col = [col for col in df_train.columns if col != 'image'][0] if len([col for col in df_train.columns if col != 'image']) > 0 else df_train.columns[0]
labels = sns.barplot(df_train[label_col].value_counts().index, df_train[label_col].value_counts())
for item in labels.get_xticklabels():
    item.set_rotation(45)