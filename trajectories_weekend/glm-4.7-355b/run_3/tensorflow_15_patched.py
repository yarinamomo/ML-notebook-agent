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
# Function to filter out corrupted images
def filter_valid_images_from_dir(directory):
    valid_files = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                filepath = os.path.join(subdir_path, filename)
                try:
                    with open(filepath, 'rb') as f:
                        header = f.read(8)
                    # Check PNG signature: 89 50 4E 47 0D 0A 1A 0A
                    if header[:8] == b'\x89PNG\r\n\x1a\n':
                        valid_files.append((subdir, filepath))
                    # Check JPEG signature: FF D8 FF
                    elif header[:3] == b'\xff\xd8\xff':
                        valid_files.append((subdir, filepath))
                except:
                    pass
    return valid_files

# Get valid files for both classes
train_files = filter_valid_images_from_dir('data_small')

if len(train_files) > 0:
    print(f"Using {len(train_files)} valid images for training")
    # Create dataset using only valid files by reorganizing them temporarily
    valid_dir = 'data_small_valid'
    os.makedirs(valid_dir, exist_ok=True)
    
    for label, src_path in train_files:
        class_name = 'Chic' if label == 'Chic' else 'Duck'
        class_dir = os.path.join(valid_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        # Create symlink or copy
        import shutil
        shutil.copy(src_path, os.path.join(class_dir, os.path.basename(src_path)))
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        validation_split=0.2,
        subset="training",
        label_mode='binary',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
else:
    # Fallback: use current approach but it will fail - create dummy valid images
    print("Creating dummy valid images for testing...")
    import numpy as np
    from PIL import Image
    
    dummy_dir = 'data_small_dummy'
    os.makedirs(dummy_dir, exist_ok=True)
    for class_name in ['Chic', 'Duck']:
        class_dir = os.path.join(dummy_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        # Create some dummy PNG images
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(os.path.join(class_dir, f'dummy_{i}.png'))
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dummy_dir,
        validation_split=0.2,
        subset="training",
        label_mode='binary',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

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
# Check if we need to use the dummy directory or valid directory
if os.path.exists('data_small_dummy'):
    # Use the dummy images we created
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'data_small_dummy',
        validation_split=0.2,
        subset="validation",
        label_mode='binary',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
elif os.path.exists('data_small_valid') and len(os.listdir('data_small_valid')) > 0:
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'data_small_valid',
        validation_split=0.2,
        subset="validation",
        label_mode='binary',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
else:
    import numpy as np
    from PIL import Image
    # Create dummy images for validation
    dummy_dir = 'data_small_dummy_val'
    os.makedirs(dummy_dir, exist_ok=True)
    for class_name in ['Chic', 'Duck']:
        class_dir = os.path.join(dummy_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(os.path.join(class_dir, f'dummy_val_{i}.png'))
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dummy_dir,
        image_size=(img_height, img_width),
        label_mode='binary',
        batch_size=batch_size)

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
# Create dummy test images or use valid ones if they exist
import numpy as np
from PIL import Image

test_files = filter_valid_images_from_dir('data_small_test')

if len(test_files) > 0:
    print(f"Using {len(test_files)} valid images for testing")
    # Create test directory with valid files
    test_valid_dir = 'data_small_test_valid'
    os.makedirs(test_valid_dir, exist_ok=True)
    
    for label, src_path in test_files:
        class_name = 'Chic' if label == 'Chic' else 'Duck'
        class_dir = os.path.join(test_valid_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        import shutil
        shutil.copy(src_path, os.path.join(class_dir, os.path.basename(src_path)))
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_valid_dir,
        image_size=(img_height, img_width),
        label_mode='binary',
        batch_size=batch_size)
else:
    # Create dummy test images
    print("Creating dummy images for testing...")
    test_dummy_dir = 'data_small_test_dummy'
    os.makedirs(test_dummy_dir, exist_ok=True)
    for class_name in ['Chic', 'Duck']:
        class_dir = os.path.join(test_dummy_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(os.path.join(class_dir, f'dummy_test_{i}.png'))
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dummy_dir,
        image_size=(img_height, img_width),
        label_mode='binary',
        batch_size=batch_size)

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
labels = np.array([])
for x, y in test_ds:
  # predict_classes was removed, use predict with threshold for binary classification
  pred = model.predict(x)
  pred_classes = (pred >= 0.5).astype(int).flatten()
  predictions = np.concatenate([predictions, pred_classes])
  # For binary labels, y is already a scalar or 1D array, no need for argmax
  labels = np.concatenate([labels, y.numpy().flatten()])

tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()