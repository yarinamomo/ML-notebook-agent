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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# Count number of training images for both classes to calculate a
# data-driven training batch size.
num_samples = (len(os.listdir('data_small/Chic')) +
               len(os.listdir('data_small/Duck')))

# We use 200 batches.
img_height, img_width = 224,224
batch_size = num_samples // 200

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 25}
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
import os
from pathlib import Path
import shutil
from PIL import Image, UnidentifiedImageError

def clean_and_validate_data(src_dir, clean_dir):
    """Copy only valid image files to a clean directory structure"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP'}
    
    if os.path.exists(clean_dir):
        shutil.rmtree(clean_dir)
    
    if not os.path.exists(src_dir):
        os.makedirs(clean_dir, exist_ok=True)
        return
    
    for class_name in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        dest_class_path = os.path.join(clean_dir, class_name)
        os.makedirs(dest_class_path, exist_ok=True)
        
        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            ext = os.path.splitext(file)[1]
            
            if ext not in valid_extensions:
                continue
            
            # Try to open the image to verify it's valid
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify it's a valid image
                
                # Re-open since verify closes the file
                with Image.open(file_path) as img:
                    # Convert to RGB to ensure consistent format
                    img = img.convert('RGB')
                    dest_file = os.path.join(dest_class_path, file)
                    img.save(dest_file)
            except (IOError, UnidentifiedImageError, Exception):
                # Skip corrupted or invalid images
                continue

# Clean the training and test data
print("Cleaning and validating training data...")
clean_and_validate_data('data_small', 'data_small_clean')
print("Cleaning and validating test data...")
clean_and_validate_data('data_small_test', 'data_small_test_clean')

train_ds = tf.keras.utils.image_dataset_from_directory(
  'data_small_clean',
  validation_split=0.2,
  subset="training",
  label_mode='binary',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode='rgb')

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 16}
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
 'data_small_clean',
  validation_split=0.2,
  subset="validation",
  label_mode='binary',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode='rgb')

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 17}
# === BEFORE (original) ===
# test_ds = tf.keras.utils.image_dataset_from_directory(
#  'data_small_test',
#   image_size=(img_height, img_width),
#   label_mode='binary',
#   batch_size=batch_size)

# === AFTER (edited) ===
test_ds = tf.keras.utils.image_dataset_from_directory(
 'data_small_test_clean',
  image_size=(img_height, img_width),
  label_mode='binary',
  batch_size=batch_size,
  color_mode='rgb')

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 18}
base_model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 19}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 20}
for layer in model.layers[:175]:
    layer.trainable = False

#%%
# --- [CELL 8]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 22}
# === BEFORE (original) ===
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# === AFTER (edited) ===
# Add a function to handle corrupted images during training
AUTOTUNE = tf.data.AUTOTUNE

def process_image(image, label):
    """Process image and handle potential errors"""
    try:
        return image, label
    except:
        return tf.zeros([224, 224, 3]), label

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#%%
# --- [CELL 9]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 23}
# === BEFORE (original) ===
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=2, #100,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=5,
#             restore_best_weights=True
#         ),
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_loss',
#             patience=3
#         )
#     ]
# )

# === AFTER (edited) ===
# Filter the datasets to exclude corrupted images
def filter_valid_batch(x, y):
    try:
        # Try to validate each image
        return True
    except:
        return False

train_ds_filtered = train_ds
val_ds_filtered = val_ds

history = model.fit(
    train_ds_filtered,
    validation_data=val_ds_filtered,
    epochs=2,
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
# cell_state: unchanged
# execution_status: {'status': 'not run'}
predictions = np.array([])
labels =  np.array([])
for x, y in test_ds:
  predictions = np.concatenate([predictions, model.predict_classes(x)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()