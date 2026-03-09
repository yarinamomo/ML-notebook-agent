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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# image_height, image_width = PIL.Image.open(str(princess[1])).size
# batch_size,epochs = 64,10

# === AFTER (edited) ===
image_width, image_height = PIL.Image.open(str(princess[1])).size
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
# First, remove or skip the corrupted file by creating a custom dataset
import glob
import os

def create_filtered_dataset(data_dir, subset, image_size, batch_size, validation_split=0.2, seed=1):
    # Get all subdirectories
    class_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(d)])
    class_names = [os.path.basename(d) for d in class_dirs]
    
    # Collect all valid image files with their labels
    all_files = []
    all_labels = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        files = glob.glob(os.path.join(class_dir, '*.jpg'))
        for file_path in files:
            # Try to open and verify the image
            try:
                with PIL.Image.open(file_path) as img:
                    img.verify()
                all_files.append(file_path)
                all_labels.append(class_idx)
            except:
                print(f"Skipping corrupted file: {file_path}")
                continue
    
    # Split into train/val
    import random
    random.seed(seed)
    indices = list(range(len(all_files)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - validation_split)) if subset == 'training' else int(len(indices) * validation_split)
    
    if subset == 'training':
        selected_indices = indices[:split_idx]
    else:
        selected_indices = indices[-split_idx:]
    
    # Create dataset
    file_paths = [all_files[i] for i in selected_indices]
    labels = [all_labels[i] for i in selected_indices]
    
    # Create tf.data.Dataset
    def load_and_preprocess(path, label):
        # Read file
        img = tf.io.read_file(path)
        # Decode image
        img = tf.image.decode_jpeg(img, channels=3)
        # Resize
        img = tf.image.resize(img, image_size)
        return img, label
    
    path_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = path_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset, class_names

train_ds, all_class_names = create_filtered_dataset(
    data_dir, 
    subset='training',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    validation_split=0.2,
    seed=1
)

val_ds, _ = create_filtered_dataset(
    data_dir,
    subset='validation', 
    image_size=(image_height, image_width),
    batch_size=batch_size,
    validation_split=0.2,
    seed=1
)

# Update class_names
train_ds.class_names = all_class_names
val_ds.class_names = all_class_names

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
# val_ds was already created in cell 5 with filtered data
# No need to recreate it

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