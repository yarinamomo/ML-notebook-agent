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
# Filter out files with unsupported/corrupt image formats before training
valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

all_image_files = [p for p in data_dir.glob('*/*') if p.is_file() and p.suffix.lower() in valid_exts]
valid_image_files = []

for p in all_image_files:
    try:
        # verify() checks file integrity/format without fully decoding into memory
        with PIL.Image.open(p) as img:
            img.verify()
        valid_image_files.append(str(p))
    except Exception:
        # skip unreadable/corrupt images
        pass

labels = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
label_to_idx = {name: i for i, name in enumerate(labels)}

def get_label(path_str):
    class_name = pathlib.Path(path_str).parent.name
    return label_to_idx[class_name]

paths_ds = tf.data.Dataset.from_tensor_slices(valid_image_files)
labels_ds = tf.data.Dataset.from_tensor_slices([get_label(p) for p in valid_image_files])
ds = tf.data.Dataset.zip((paths_ds, labels_ds))

# shuffle/split (same 80/20 strategy)
size = len(valid_image_files)
val_size = int(0.2 * size)
train_size = size - val_size

ds = ds.shuffle(size, seed=1, reshuffle_each_iteration=False)
train_raw = ds.take(train_size)
val_raw = ds.skip(train_size)

def decode_and_resize(path, label):
    img = tf.io.read_file(path)
    # decode_image can handle jpg/png/gif/bmp; set channels=3 for consistency
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_height, image_width])
    return img, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_raw.map(decode_and_resize, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
val_ds = val_raw.map(decode_and_resize, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

history = model.fit(train_ds, validation_data=val_ds, epochs=1)