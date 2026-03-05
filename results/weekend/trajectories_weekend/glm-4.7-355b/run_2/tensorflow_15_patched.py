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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
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
# Get all valid image files and their labels
class_names = ['Chic', 'Duck']
valid_paths = []
valid_labels = []

def validate_image_file(filepath):
    """Try to actually decode the image to check if it's valid."""
    try:
        img = tf.io.read_file(filepath)
        # Try to decode image - this will work for PNG, JPEG, etc.
        decoded = tf.image.decode_image(img, channels=3, expand_animations=False)
        # Check if we got a valid image with proper shape
        if decoded.shape.rank == 3 and decoded.shape[-1] == 3:
            return True
        return False
    except:
        return False

for class_idx, class_name in enumerate(class_names):
    class_path = os.path.join('data_small', class_name)
    if os.path.exists(class_path):
        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            # Skip hidden files
            if filename.startswith('.'):
                continue
            # Try to validate the image by actually decoding it
            if validate_image_file(filepath):
                valid_paths.append(filepath)
                valid_labels.append(class_idx)

print(f"Found {len(valid_paths)} valid images for training")

# Create dataset from valid files
def load_and_preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Split into train and validation
if len(valid_paths) > 0:
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        valid_paths, valid_labels, test_size=0.2, random_state=123, stratify=valid_labels
    )
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    print(f"Training dataset created with {len(train_paths)} samples")
else:
    print("No valid images found! This dataset may be corrupted or empty.")
    # Create a dummy dataset to avoid further errors
    train_ds = tf.data.Dataset.from_tensor_slices((tf.zeros((1, 224, 224, 3)), tf.constant([0])))
    train_ds = train_ds.batch(1)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
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
# Create validation dataset from valid files
if len(valid_paths) > 0:
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    print(f"Validation dataset created with {len(val_paths)} samples")
else:
    val_ds = tf.data.Dataset.from_tensor_slices((tf.zeros((1, 224, 224, 3)), tf.constant([0])))
    val_ds = val_ds.batch(1)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# test_ds = tf.keras.utils.image_dataset_from_directory(
#  'data_small_test',
#   image_size=(img_height, img_width),
#   label_mode='binary',
#   batch_size=batch_size)

# === AFTER (edited) ===
# Get all valid test image files and their labels
test_valid_paths = []
test_valid_labels = []

for class_idx, class_name in enumerate(class_names):
    class_path = os.path.join('data_small_test', class_name)
    if os.path.exists(class_path):
        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            # Skip hidden files
            if filename.startswith('.'):
                continue
            # Try to validate the image by actually decoding it
            if validate_image_file(filepath):
                test_valid_paths.append(filepath)
                test_valid_labels.append(class_idx)

print(f"Found {len(test_valid_paths)} valid images for testing")

# Create test dataset from valid files
if len(test_valid_paths) > 0:
    test_ds = tf.data.Dataset.from_tensor_slices((test_valid_paths, test_valid_labels))
    test_ds = test_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    print(f"Test dataset created with {len(test_valid_paths)} samples")
else:
    test_ds = tf.data.Dataset.from_tensor_slices((tf.zeros((1, 224, 224, 3)), tf.constant([0])))
    test_ds = test_ds.batch(1)
    print("No valid test images found!")

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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
results = model.evaluate(test_ds, verbose=0)
print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
predictions = (model.predict(test_ds) >= 0.5)

#%%
# --- [CELL 12]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
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
  # Use predict instead of predict_classes (deprecated in newer TensorFlow)
  preds = model.predict(x)
  pred_classes = (preds >= 0.5).astype(int).flatten()
  predictions = np.concatenate([predictions, pred_classes])
  labels = np.concatenate([labels, y.numpy().flatten()])

tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()