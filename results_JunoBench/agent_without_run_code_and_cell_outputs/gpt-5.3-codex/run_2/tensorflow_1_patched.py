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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# import pathlib
# data_dir = 'data/web_scraped_small'
# data_dir = pathlib.Path(data_dir).with_suffix('')

# === AFTER (edited) ===
import pathlib
data_dir = 'data/web_scraped_small'
data_dir = pathlib.Path(data_dir).with_suffix('')

# Keep only image files TensorFlow can decode
valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
for p in data_dir.rglob('*'):
    if p.is_file() and p.suffix.lower() not in valid_exts:
        p.unlink()

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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    image_size=(image_height, image_width),
    seed = 1,
    shuffle=True,
    batch_size=batch_size
)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    image_size=(image_height, image_width),
    seed = 1,
    shuffle=True,
    batch_size=batch_size
)

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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
# === BEFORE (original) ===
# history = model.fit(train_ds,validation_data=val_ds, epochs=1)

# === AFTER (edited) ===
# Rebuild datasets, filtering out files that fail image decoding
all_files = tf.io.gfile.glob(str(data_dir / '*' / '*'))
class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
class_to_idx = {name: i for i, name in enumerate(class_names)}

valid_paths, valid_labels = [], []
for fp in all_files:
    cls = pathlib.Path(fp).parent.name
    if cls not in class_to_idx:
        continue
    try:
        b = tf.io.read_file(fp)
        _ = tf.io.decode_image(b, channels=3, expand_animations=False)
        valid_paths.append(fp)
        valid_labels.append(class_to_idx[cls])
    except Exception:
        pass

paths_ds = tf.data.Dataset.from_tensor_slices(valid_paths)
labels_ds = tf.data.Dataset.from_tensor_slices(valid_labels)

def load_and_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_height, image_width])
    return tf.cast(img, tf.float32), label

ds = tf.data.Dataset.zip((paths_ds, labels_ds)).shuffle(len(valid_paths), seed=1)
ds = ds.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

val_size = int(0.2 * len(valid_paths))
val_ds = ds.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
train_ds = ds.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

history = model.fit(train_ds, validation_data=val_ds, epochs=1)