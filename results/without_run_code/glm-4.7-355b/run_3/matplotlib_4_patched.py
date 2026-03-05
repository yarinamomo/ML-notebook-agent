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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
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

# Add file extension filter to only accept valid image formats
valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff'}

def count_files_by_extension(directory):
    """Count files by extension in directory"""
    ext_count = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            ext_count[ext] = ext_count.get(ext, 0) + 1
    return ext_count

import os
try:
    ext_counts = count_files_by_extension(data_dir)
    print(f"File extensions in dataset: {ext_counts}")
except:
    print("Could not count file extensions")

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

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# class_names = train_ds.class_names
# print(class_names)

# === AFTER (edited) ===
class_names = train_ds.class_names
print(class_names)

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
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

# Try to display images, but handle corrupted data gracefully
try:
    # Attempt to iterate through the dataset multiple times to find valid batches
    displayed_count = 0
    max_display = 6
    
    for batch_num in range(20):  # Try up to 20 batches
        if displayed_count >= max_display:
            break
        
        try:
            # Try to get a batch from the dataset
            for images, labels in train_ds.skip(batch_num).take(1):
                for i in range(len(images)):
                    if displayed_count >= max_display:
                        break
                    try:
                        ax = plt.subplot(2, 3, displayed_count + 1)
                        plt.imshow(images[i].numpy().astype("uint8"))
                        # label is one-hot encoded, get the index
                        label_index = tf.argmax(labels[i]).numpy()
                        plt.title(class_names[label_index])
                        plt.axis("off")
                        displayed_count += 1
                    except Exception as inner_e:
                        # Skip individual corrupted images
                        continue
        except Exception as batch_e:
            # Skip corrupted batches
            continue
    
    if displayed_count == 0:
        print("Could not load any images from the dataset due to corrupted files.")
    else:
        print(f"Successfully displayed {displayed_count} images.")
        
except Exception as e:
    print(f"Error displaying images: {e}")
    print("The dataset may contain corrupted image files.")