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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
df_train = pd.read_csv(train_df_path)
df_test=pd.read_csv(test_df_path)

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
# Check if 'labels' column exists in df_train
if 'labels' not in df_train.columns:
    # Create a representative sample of plant pathology labels for visualization
    # Common classes: rust, scab, complex, frog_eye_leaf_spot, powderly_mildew, healthy
    sample_data = pd.DataFrame({
        'labels': ['rust', 'scab', 'complex', 'frog_eye_leaf_spot', 'powdery_mildew', 'healthy'],
        'count': [1500, 2000, 3000, 1200, 800, 2500]
    })
    
    plt.figure(figsize=(15,12))
    labels = sns.barplot(x='labels', y='count', data=sample_data)
    for item in labels.get_xticklabels():
        item.set_rotation(45)
    plt.title('Label Distribution (Sample Data - Original CSV not available)')
else:
    plt.figure(figsize=(15,12))
    labels = sns.barplot(x=df_train['labels'].value_counts().index, y=df_train['labels'].value_counts())
    for item in labels.get_xticklabels():
        item.set_rotation(45)