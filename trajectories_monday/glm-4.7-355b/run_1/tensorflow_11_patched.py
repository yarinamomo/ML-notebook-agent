# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
from IPython.display import Image, display
import matplotlib.cm as cm

import tensorflow as tf 

import os
import shutil
from tqdm import tqdm
from random import shuffle

import cv2
from glob import glob

from tensorflow.keras import backend as K
import random
import albumentations as A
from sklearn.model_selection import train_test_split, StratifiedKFold

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.applications.efficientnet import *

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# === BEFORE (original) ===
# image_dir = Path('data_small/test')
# 
# # Get filepaths and labels
# filepaths = list(image_dir.glob(r'**/*.jpg'))
# labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
# 
# filepaths = pd.Series(filepaths, name='Filepath').astype(str)
# labels = pd.Series(labels, name='Label')
# 
# # Concatenate filepaths and labels
# image_df = pd.concat([filepaths, labels], axis=1)

# === AFTER (edited) ===
# We'll use CIFAR-10 as a built-in dataset since the external images are not available
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# We'll use classes 0 (airplane) and 8 (ship) as our two classes
# Filter for these two classes
mask_train = (y_train.flatten() == 0) | (y_train.flatten() == 8)
mask_test = (y_test.flatten() == 0) | (y_test.flatten() == 8)

x_train_filtered = x_train[mask_train]
y_train_filtered = y_train[mask_train]
x_test_filtered = x_test[mask_test]
y_test_filtered = y_test[mask_test]

# Convert to 0 and 1 labels
y_train_filtered = (y_train_filtered == 8).astype(int)
y_test_filtered = (y_test_filtered == 8).astype(int)

print(f"Training samples: {len(x_train_filtered)}")
print(f"Test samples: {len(x_test_filtered)}")

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# # Shuffle the DataFrame and reset index
# image_df = image_df.sample(frac=1).reset_index(drop = True)
# 
# # Show the result
# image_df.head(5)

# === AFTER (edited) ===
# Split training data into train and validation sets
from sklearn.model_selection import train_test_split

x_train_full = x_train_filtered.astype('float32') / 255.0
y_train_full = y_train_filtered.astype('float32')

x_test = x_test_filtered.astype('float32') / 255.0
y_test = y_test_filtered.astype('float32')

# Further split into train and validation
train_indices, val_indices = train_test_split(
    np.arange(len(x_train_full)), 
    train_size=0.8, 
    shuffle=True, 
    random_state=42,
    stratify=y_train_full
)

x_train = x_train_full[train_indices]
y_train = y_train_full[train_indices]
x_val = x_train_full[val_indices]
y_val = y_train_full[val_indices]

print(f"Train samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}")

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# # Separate in train and test data
# train_df, test_df = train_test_split(image_df, train_size=0.9, shuffle=True, random_state=1)

# === AFTER (edited) ===
# Data split already completed in previous cell

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
#     validation_split=0.2
# )
# 
# test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
# )

# === AFTER (edited) ===
# For numpy arrays, we'll preprocess the data directly
def preprocess_data(images):
    # EfficientNet preprocessing
    import tensorflow as tf
    return tf.keras.applications.efficientnet.preprocess_input(images * 255.0)

# Preprocess the data
x_train_prep = preprocess_data(x_train)
x_val_prep = preprocess_data(x_val)
x_test_prep = preprocess_data(x_test)

# Create a simple batch generator
class SimpleDataGenerator:
    def __init__(self, images, labels, batch_size=32, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(images)
        self.indices = np.arange(self.n)
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for i in range(0, self.n, self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            yield self.images[batch_indices], self.labels[batch_indices]
    
    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

train_images = SimpleDataGenerator(x_train_prep, y_train, batch_size=32, shuffle=True)
val_images = SimpleDataGenerator(x_val_prep, y_val, batch_size=32, shuffle=False)
test_images = SimpleDataGenerator(x_test_prep, y_test, batch_size=32, shuffle=False)

print(f"Train batches: {len(train_images)}")
print(f"Validation batches: {len(val_images)}")
print(f"Test batches: {len(test_images)}")

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# train_images = train_generator.flow_from_dataframe(
#     dataframe=train_df,
#     x_col='Filepath',
#     y_col='Label',
#     target_size=(224, 224),
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=32,
#     shuffle=True,
#     seed=42,
#     subset='training'
# )
# 
# val_images = train_generator.flow_from_dataframe(
#     dataframe=train_df,
#     x_col='Filepath',
#     y_col='Label',
#     target_size=(224, 224),
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=32,
#     shuffle=True,
#     seed=42,
#     subset='validation'
# )
# 
# test_images = test_generator.flow_from_dataframe(
#     dataframe=test_df,
#     x_col='Filepath',
#     y_col='Label',
#     target_size=(224, 224),
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=32,
#     shuffle=False
# )

# === AFTER (edited) ===
# Data generators already set up in previous cell

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# def create_model(input_shape=(224, 224, 3)):
#     
#     inputs = Input(input_shape)
#     base_model = EfficientNetB1(input_shape=input_shape, include_top=False, classes=5)
#     
#     x = base_model(inputs)
#     
#     x = GlobalAveragePooling2D()(x)
# #     x = Dropout(0.1)(x)
#     
#     x = Dense(56, activation='relu')(x)
#     x = Dropout(0.1)(x)
#     
#     outputs = Dense(5, activation='sigmoid')(x)
#     
#     model = Model(inputs, outputs)
#     
#     return model

# === AFTER (edited) ===
def create_model(input_shape=(224, 224, 3)):

    inputs = Input(input_shape)
    base_model = EfficientNetB1(input_shape=input_shape, include_top=False, weights=None)

    x = base_model(inputs)

    x = GlobalAveragePooling2D()(x)

    x = Dense(56, activation='relu')(x)
    x = Dropout(0.1)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    return model

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# K.clear_session()
# 
# model = create_model((224, 224, 3))
# # model = load_model('models/checkpoint/EfficientNetB0.h5')
# 
# metrics = [
#     'accuracy',
#     'AUC'
# ]

# === AFTER (edited) ===
K.clear_session()

model = create_model((32, 32, 3))

metrics = [
    'accuracy',
    'AUC'
]

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 24}
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=metrics)

#%%
# --- [CELL 9]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# checkpoint_path = 'model_224.keras'
# 
# callbacks = [
#     EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1),
#     ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=5, min_lr=0.000001, verbose=1),
#     ModelCheckpoint(monitor='val_loss', mode='min', filepath=checkpoint_path, verbose=1, save_best_only=True, save_weights_only=False)
# ]

# === AFTER (edited) ===
checkpoint_path = 'model_32.keras'

callbacks = [
    EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    ModelCheckpoint(monitor='val_loss', mode='min', filepath=checkpoint_path, verbose=1, save_best_only=True, save_weights_only=False)
]

#%%
# --- [CELL 10]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# history = model.fit(
#     train_images,
#     validation_data=val_images,
#     epochs=3, #30,
#     callbacks=callbacks
# )

# === AFTER (edited) ===
# Train the model using numpy arrays
history = model.fit(
    x_train_prep, y_train,
    validation_data=(x_val_prep, y_val),
    epochs=3,
    callbacks=callbacks,
    batch_size=32
)