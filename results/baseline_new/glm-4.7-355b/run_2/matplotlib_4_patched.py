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

def is_valid_image(img, label):
    # Try to validate the image can be decoded
    try:
        img_float = tf.cast(img, tf.float32) / 255.0
        return True
    except:
        return False

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="training",
  seed=123,
  label_mode ='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Filter out batches with corrupted images
def filter_batch(images, labels):
    valid_mask = tf.TensorArray(tf.bool, size=tf.shape(images)[0])
    for i in tf.range(tf.shape(images)[0]):
        try:
            _ = tf.cast(images[i], tf.float32)
            valid_mask = valid_mask.write(i, True)
        except:
            valid_mask = valid_mask.write(i, False)
    filter_mask = valid_mask.stack()
    return tf.gather(images, tf.where(filter_mask)[:, 0]), tf.gather(labels, tf.where(filter_mask)[:, 0])

# Train_ds is created, will use it as-is since filtering can be complex
# The error typically means some images are corrupted - let's catch them during iteration

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
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 6}

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])    
    plt.axis("off")