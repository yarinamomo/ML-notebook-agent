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
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    image_size=(image_height, image_width),
    seed=1,
    shuffle=True,
    batch_size=batch_size,
    label_mode='int',
    interpolation='bilinear',
    follow_links=False
)

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
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    image_size=(image_height, image_width),
    seed=1,
    shuffle=True,
    batch_size=batch_size,
    label_mode='int',
    interpolation='bilinear',
    follow_links=False
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
def _valid_image(path):
    try:
        img = tf.io.read_file(path)
        tf.io.decode_image(img, channels=3, expand_animations=False)
        return True
    except Exception:
        return False

all_files = tf.io.gfile.glob(str(data_dir / '*' / '*'))
valid_files = [p for p in all_files if _valid_image(p)]

labels = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
label_to_index = {name: i for i, name in enumerate(labels)}
valid_labels = [label_to_index[pathlib.Path(p).parent.name] for p in valid_files]

train_size = int(0.8 * len(valid_files))
train_files, val_files = valid_files[:train_size], valid_files[train_size:]
train_labels, val_labels = valid_labels[:train_size], valid_labels[train_size:]

train_ds_clean = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
val_ds_clean = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

def _load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [image_height, image_width])
    return img, label

train_ds_clean = train_ds_clean.map(_load_img).batch(batch_size)
val_ds_clean = val_ds_clean.map(_load_img).batch(batch_size)

history = model.fit(train_ds_clean, validation_data=val_ds_clean, epochs=1)