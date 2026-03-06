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
# execution_status: {'status': 'error', 'done': True, 'execution_count': 6}
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
import tensorflow as tf

# Add preprocessing to normalize images and handle edge cases
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Preprocess the datasets
train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Get class names (available on the original dataset object from image_dataset_from_directory)
class_names = train_ds.class_names
print(f"Number of classes: {len(class_names)}")

plt.figure(figsize=(10, 10))
try:
    for images, labels in train_ds.take(1):
        # Convert one-hot encoded labels to indices
        label_indices = tf.argmax(labels, axis=1)
        
        # Use min of 6 and batch size
        num_images = min(6, images.shape[0])
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            # Convert normalized floats back to uint8 for display
            img = (images[i].numpy() * 255).astype("uint8")
            plt.imshow(img)
            class_idx = label_indices[i].numpy()
            if class_idx < len(class_names):
                plt.title(class_names[class_idx])
            else:
                plt.title(f"Class {class_idx}")
            plt.axis("off")
except Exception as e:
    print(f"Error during visualization: {e}")
    # Try to show a simplified plot if full visualization fails
    print("Dataset may contain some invalid image files.")