# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np # linear algebra
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from matplotlib import pyplot as plt

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
import pathlib
data_dir = 'data/web_scraped_small'
data_dir = pathlib.Path(data_dir).with_suffix('')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
image_count = len(list(data_dir.glob('*/*.jpg')))
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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
image_height, image_width = PIL.Image.open(str(princess[1])).size
batch_size,epochs = 64,10

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
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
def load_and_decode_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_height, image_width])
    img.set_shape([None, None, 3])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize_with_crop_or_pad(img, image_height, image_width)
    img.set_shape([image_height, image_width, 3])
    return img, label

def get_class_names():
    class_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
    return class_names

all_file_paths = []
all_labels = []
class_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())

for i, class_name in enumerate(class_names):
    class_path = data_dir / class_name
    for img_path in class_path.glob('*.jpg'):
        all_file_paths.append(str(img_path))
        all_labels.append(i)

valid_indices = []
for i, path in enumerate(all_file_paths):
    try:
        img_bytes = tf.io.read_file(path)
        decoded = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        valid_indices.append(i)
    except:
        pass

print(f"Valid images: {len(valid_indices)} out of {len(all_file_paths)}")

valid_file_paths = [all_file_paths[i] for i in valid_indices]
valid_labels = [all_labels[i] for i in valid_indices]

full_ds = tf.data.Dataset.from_tensor_slices((valid_file_paths, valid_labels))
full_ds = full_ds.map(load_and_decode_image, num_parallel_calls=tf.data.AUTOTUNE)

train_size = int(0.8 * len(valid_file_paths))
val_size = len(valid_file_paths) - train_size

full_ds = full_ds.shuffle(buffer_size=len(valid_file_paths), seed=42)
train_ds = full_ds.take(train_size).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
val_ds = full_ds.skip(train_size).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

print(f"Training samples: {train_size}, Validation samples: {val_size}")

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
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
# Validation dataset already created in cell 5
pass

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
normalization_layer = layers.Rescaling(1./255)

#%%
# --- [CELL 8]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# from tensorflow import keras
# data_augmentation = keras.Sequential(
#   [
#     layers.RandomFlip("horizontal",
#                       input_shape=(image_height,
#                                   image_width,
#                                   3)),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#   ]
# )

# === AFTER (edited) ===
from tensorflow import keras
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

#%%
# --- [CELL 9]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# num_of_classes = len(train_ds.class_names)
# num_of_classes

# === AFTER (edited) ===
def get_class_names():
    class_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
    return class_names

class_names = get_class_names()
num_of_classes = len(class_names)
num_of_classes

#%%
# --- [CELL 10]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# model = Sequential([
#   data_augmentation,
#   normalization_layer,
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Dropout(0.2),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_of_classes, name="outputs")
# ])

# === AFTER (edited) ===
model = Sequential([
  layers.Input(shape=(image_height, image_width, 3)),
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
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 13}
# === BEFORE (original) ===
# history = model.fit(train_ds,validation_data=val_ds, epochs=1)

# === AFTER (edited) ===
history = model.fit(train_ds, validation_data=val_ds, epochs=1)