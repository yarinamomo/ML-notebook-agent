# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
from pathlib import Path
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import subprocess
from IPython.display import FileLink, display

from sklearn.metrics import confusion_matrix, classification_report

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# # Count number of training images for both classes to calculate a
# # data-driven training batch size.
# num_samples = (len(os.listdir('data_small/Chic')) +
#                len(os.listdir('data_small/Duck')))
# 
# # We use 200 batches.
# img_height, img_width = 224,224
# batch_size = num_samples // 200

# === AFTER (edited) ===
from PIL import Image

def is_valid_image(filepath):
    """Check if an image file is valid by trying to open it"""
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verify that it is a valid image
        # Reopen as verify closes the file
        with Image.open(filepath) as img:
            img.load()
        return True
    except Exception as e:
        print(f"Invalid image: {filepath} - {e}")
        return False

def clean_corrupted_images(directory):
    """Remove corrupted images from the directory"""
    removed = []
    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            if os.path.isfile(filepath):
                if not is_valid_image(filepath):
                    try:
                        os.remove(filepath)
                        removed.append(filepath)
                    except:
                        pass
    
    if removed:
        print(f"Removed {len(removed)} corrupted files from {directory}")
    else:
        print(f"No corrupted files found in {directory}")
    return len(removed)

# Clean the datasets
print("Cleaning datasets...")
clean_corrupted_images('data_small')
clean_corrupted_images('data_small_test')

# Recalculate num_samples after cleaning
num_samples = (len(os.listdir('data_small/Chic')) +
               len(os.listdir('data_small/Duck')))

img_height, img_width = 224,224
batch_size = max(1, num_samples // 200)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# train_ds = tf.keras.utils.image_dataset_from_directory(
#   'data_small',
#   validation_split=0.2,
#   subset="training",
#   label_mode='binary',
#   seed=123, #number to randomize outcome
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# === AFTER (edited) ===
train_ds = tf.keras.utils.image_dataset_from_directory(
  'data_small',
  validation_split=0.2,
  subset="training",
  label_mode='binary',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  ignore_corrupted_files=True)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# val_ds = tf.keras.utils.image_dataset_from_directory(
#  'data_small',
#   validation_split=0.2,
#   subset="validation",
#   label_mode='binary',
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# === AFTER (edited) ===
val_ds = tf.keras.utils.image_dataset_from_directory(
 'data_small',
  validation_split=0.2,
  subset="validation",
  label_mode='binary',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  ignore_corrupted_files=True)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# test_ds = tf.keras.utils.image_dataset_from_directory(
#  'data_small_test',
#   image_size=(img_height, img_width),
#   label_mode='binary',
#   batch_size=batch_size)

# === AFTER (edited) ===
test_ds = tf.keras.utils.image_dataset_from_directory(
 'data_small_test',
  image_size=(img_height, img_width),
  label_mode='binary',
  batch_size=batch_size,
  ignore_corrupted_files=True)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
base_model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)

# 1024 neurons are half the 2048 output dimensionality of the previous
# layer and the last layer from base ResNet-50.
x = keras.layers.Dense(units=1024, activation='relu')(x)
x = keras.layers.Dense(units=512, activation='relu')(x)
x = keras.layers.Dense(units=256, activation='relu')(x)
x = keras.layers.Dense(units=128, activation='relu')(x)
x = keras.layers.Dense(units=1, activation='sigmoid')(x)

model = keras.models.Model(inputs=base_model.input,
                                    outputs=x)

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
for layer in model.layers[:175]:
    layer.trainable = False

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 10}
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2, #100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=3
        )
    ]
)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
results = model.evaluate(test_ds, verbose=0)
print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
predictions = (model.predict(test_ds) >= 0.5)

#%%
# --- [CELL 12]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# predictions = np.array([])
# labels =  np.array([])
# for x, y in test_ds:
#   predictions = np.concatenate([predictions, model.predict_classes(x)])
#   labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
# 
# tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()

# === AFTER (edited) ===
predictions = np.array([])
labels =  np.array([])
for x, y in test_ds:
    pred = (model.predict(x) >= 0.5).astype(int).flatten()
    predictions = np.concatenate([predictions, pred])
    labels = np.concatenate([labels, y.numpy().flatten()])

tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()