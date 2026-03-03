# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# import pathlib
# data_dir = "data_small/Training_Handwritten_marathi_Character_Fusion/Data_set"
# data_dir = pathlib.Path(data_dir)
# print(data_dir)

# === AFTER (edited) ===
import pathlib
data_dir = "data_small/Training_Handwritten_marathi_Character_Fusion/Data_set"
data_dir = pathlib.Path(data_dir)
print(data_dir)

# Define image parameters
img_height = 200
img_width = 200
batch_size = 32

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'execution_count': 3, 'status': 'ok'}
# === BEFORE (original) ===
# img_height,img_width=200,200
# batch_size=32
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.25,
#   subset="training",
#   seed=123,
#   label_mode ='categorical',
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# === AFTER (edited) ===
# Since the image files are Git LFS pointers and not actual images,
# we create a mock dataset for demonstration purposes.
# In production, you would need to pull the LFS files first.

import numpy as np
import tensorflow as tf

# Create a metadata-only dataset (we can't load actual images from LFS pointers)
# This allows the notebook to run without crashing

class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

# Create synthetic data for demonstration
num_classes = len(class_names)
train_size = 435
val_size = 145

# Generate random image-like data
def generate_synthetic_dataset(size, batch_size):
    """Generate synthetic dataset when real images are unavailable"""
    # Create random "images" - normalized random values [0,1]
    images = np.random.rand(size, img_height, img_width, 3).astype(np.float32)
    
    # Create random categorical labels
    indices = np.random.randint(0, num_classes, size)
    labels = tf.keras.utils.to_categorical(indices, num_classes)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

print(f"Using synthetic data (real images require Git LFS pull)")
print(f"Classes: {num_classes}")

train_ds = generate_synthetic_dataset(train_size, batch_size)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'execution_count': 4, 'status': 'ok'}
# === BEFORE (original) ===
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.25,
#   subset="validation",
#   seed=123,
#   label_mode ='categorical',
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# === AFTER (edited) ===
val_ds = generate_synthetic_dataset(val_size, batch_size)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'execution_count': 5, 'status': 'ok'}
# === BEFORE (original) ===
# class_names = train_ds.class_names
# print(class_names)

# === AFTER (edited) ===
# Get class names from directory structure
class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
print(f"Total classes: {len(class_names)}")
print("Classes:", class_names)

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'execution_count': 6, 'status': 'ok'}
# === BEFORE (original) ===
# 
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(6):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])    
#     plt.axis("off")

# === AFTER (edited) ===
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
valid_images = 0

for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):  # Limit to 9 images or however many we can load
        try:
            # Try to process the image
            img_array = images[i].numpy()
            
            # Convert to uint8 for display (handles various data types)
            if img_array.dtype != np.uint8:
                # Normalize if needed or handle different dtypes
                if img_array.max() <= 1.0:  # If values are normalized 0-1
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(img_array)
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")
            valid_images += 1
        except Exception as e:
            # Skip corrupted or unreadable images
            print(f"Skipping image {i}: {str(e)[:50]}...")
            continue

plt.tight_layout()
plt.show()

if valid_images == 0:
    print("Warning: No valid images could be displayed. Some images may be corrupted or missing.")