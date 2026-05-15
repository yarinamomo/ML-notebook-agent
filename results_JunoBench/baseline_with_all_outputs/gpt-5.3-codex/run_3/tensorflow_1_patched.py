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
# Filter out files that are not valid images, then rebuild datasets and train
valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
all_files = sorted([p for p in data_dir.glob('*/*') if p.is_file()])

valid_paths, valid_labels = [], []
for p in all_files:
    if p.suffix.lower() not in valid_exts:
        continue
    try:
        with PIL.Image.open(p) as img:
            img.verify()
        valid_paths.append(str(p))
        valid_labels.append(p.parent.name)
    except Exception:
        # skip corrupted or unsupported files
        pass

class_names = sorted(set(valid_labels))
class_to_idx = {name: i for i, name in enumerate(class_names)}
label_indices = np.array([class_to_idx[lbl] for lbl in valid_labels], dtype=np.int32)

# deterministic split (same ratio as before)
rng = np.random.default_rng(1)
idx = np.arange(len(valid_paths))
rng.shuffle(idx)
split = int(0.8 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

train_paths = np.array(valid_paths)[train_idx]
train_labels = label_indices[train_idx]
val_paths = np.array(valid_paths)[val_idx]
val_labels = label_indices[val_idx]

def decode_and_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_height, image_width])
    img.set_shape((image_height, image_width, 3))
    return img, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds_clean = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds_clean = train_ds_clean.map(decode_and_resize, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

val_ds_clean = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds_clean = val_ds_clean.map(decode_and_resize, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

history = model.fit(train_ds_clean, validation_data=val_ds_clean, epochs=1)