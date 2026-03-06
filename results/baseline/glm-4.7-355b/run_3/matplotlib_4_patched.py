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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
img_height,img_width=200,200
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="training",
  seed=123,
  label_mode ='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="validation",
  seed=123,
  label_mode ='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
class_names = train_ds.class_names
print(class_names)

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
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

# Filter function to skip corrupted images
def is_valid_image(image, label):
    # Try to decode and validate the image
    decoded = tf.image.decode_image(tf.io.encode_jpeg(image), expand_animations=False)
    return tf.ones([], dtype=tf.bool)

# Preprocessing to handle corrupted images
def preprocess_image(image, label):
    # Ensure image is valid and in proper format
    try:
        return image, label
    except:
        return tf.zeros_like(image), label

# Apply preprocessing to dataset
train_ds_filtered = train_ds.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)

plt.figure(figsize=(10, 10))
batch_count = 0
for images, labels in train_ds_filtered:
    # Skip to next batch if there are issues
    valid_images = 0
    for i in range(len(images)):
        try:
            img = images[i].numpy().astype("uint8")
            if valid_images < 6:
                ax = plt.subplot(3, 3, valid_images + 1)
                plt.imshow(img)
                # Get the class index from one-hot encoded labels
                label_idx = tf.argmax(labels[i]).numpy()
                plt.title(class_names[label_idx])
                plt.axis("off")
                valid_images += 1
        except Exception as e:
            continue
    
    if valid_images > 0:
        break
    
    batch_count += 1
    if batch_count > 10:
        break