# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
import pathlib
data_dir = "data_small/Training_Handwritten_marathi_Character_Fusion/Data_set"
data_dir = pathlib.Path(data_dir)
print(data_dir)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
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
img_height,img_width=200,200
batch_size=32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="training",
  seed=123,
  label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Filter out corrupted batches
train_ds = train_ds.ignore_errors()

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
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
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="validation",
  seed=123,
  label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Filter out corrupted batches
val_ds = val_ds.ignore_errors()

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 5}
class_names = train_ds.class_names
print(class_names)

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 18}
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
batch_count = 0
for images, labels in train_ds.take(10):  # Try more batches until we get enough images
    for i in range(len(images)):
        if batch_count >= 6:  # We need 6 images
            break
        try:
            ax = plt.subplot(3, 3, batch_count + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            # Find the class index from the one-hot encoded label
            label_idx = np.argmax(labels[i].numpy())
            plt.title(class_names[label_idx])
            plt.axis("off")
            batch_count += 1
        except Exception as e:
            print(f"Skipping corrupted image: {e}")
            continue
    if batch_count >= 6:
        break