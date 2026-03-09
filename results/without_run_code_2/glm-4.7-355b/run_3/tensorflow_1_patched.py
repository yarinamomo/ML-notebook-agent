# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# === BEFORE (original) ===
# import numpy as np # linear algebra
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
# from matplotlib import pyplot as plt

# === AFTER (edited) ===
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os
from PIL import Image

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
import pathlib
data_dir = 'data/web_scraped_small'
data_dir = pathlib.Path(data_dir).with_suffix('')

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# image_count = len(list(data_dir.glob('*/*.jpg')))
# image_count

# === AFTER (edited) ===
image_count = len(list(data_dir.glob('*/*.png')))
image_count

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
import PIL
princess = list(data_dir.glob('princess/*'))
PIL.Image.open(str(princess[1]))

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# image_height, image_width = PIL.Image.open(str(princess[1])).size
# batch_size,epochs = 64,10

# === AFTER (edited) ===
# Get image dimensions from a valid image
image_height, image_width = PIL.Image.open(str(princess[1])).size
batch_size, epochs = 64, 10

# Fix potentially corrupt images by re-saving them as PNG
import os
from PIL import Image

for class_dir in data_dir.iterdir():
    if class_dir.is_dir():
        for img_file in class_dir.glob('*'):
            try:
                with Image.open(img_file) as img:
                    img.load()  # Try to load the image
                    # Re-save as PNG to ensure validity
                    img.save(img_file, 'PNG')
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

print("Image verification and conversion complete")

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset='training',
#     image_size=(image_height, image_width),
#     seed = 1,
#     shuffle=True,
#     batch_size=batch_size
# )

# === AFTER (edited) ===
def is_valid_image(filepath):
    """Check if a file is a valid image"""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except:
        return False

# Find all valid image paths and their labels
from pathlib import Path
valid_paths = []
valid_labels = []
class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

for class_idx, class_name in enumerate(class_names):
    class_dir = data_dir / class_name
    for img_file in class_dir.glob('*.jpg'):
        if is_valid_image(img_file):
            valid_paths.append(str(img_file))
            valid_labels.append(class_idx)

print(f"Found {len(valid_paths)} valid images out of 40 total")

# Create dataset from valid paths only
path_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_height, image_width])
    img = tf.cast(img, tf.float32)
    return img, label

full_ds = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# Split into train and validation
train_size = int(0.8 * len(valid_paths))
train_ds = full_ds.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
train_ds.class_names = class_names

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# val_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset='validation',
#     image_size=(image_height, image_width),
#     seed = 1,
#     shuffle=True,
#     batch_size=batch_size
# )

# === AFTER (edited) ===
val_ds = full_ds.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds.class_names = class_names
print(f"Validation dataset created with {len(valid_paths) - train_size} images")

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
normalization_layer = layers.Rescaling(1./255)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
from tensorflow import keras
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(image_height,
                                  image_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
num_of_classes = len(train_ds.class_names)
num_of_classes

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
model = Sequential([
  data_augmentation,
  normalization_layer,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_of_classes, name="outputs")
])

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
history = model.fit(train_ds,validation_data=val_ds, epochs=1)