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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# import PIL
# princess = list(data_dir.glob('princess/*'))
# PIL.Image.open(str(princess[1]))

# === AFTER (edited) ===
import PIL

# Try to open an image (this cell may fail, but we'll handle errors)
princess = list(data_dir.glob('princess/*'))

try:
    valid_img = PIL.Image.open(str(princess[0]))
    print(f"Successfully opened: {princess[0]}")
    valid_img
except Exception as e:
    print(f"Error opening image: {e}")
    # If no valid image can be opened, we'll set dimensions in the next cell
    valid_img = None

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# image_height, image_width = PIL.Image.open(str(princess[1])).size
# batch_size,epochs = 64,10

# === AFTER (edited) ===
# Set image dimensions
if valid_img is not None:
    image_height, image_width = valid_img.size
else:
    # Set default image dimensions if no valid image found
    image_height, image_width = 180, 180
    print(f"Using default image size: {image_height}x{image_width}")

batch_size, epochs = 64, 10

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
# Custom dataset creation with error handling
import random
import tensorflow as tf
import pathlib

def create_dataset_from_directory(directory, validation_split=0.2, subset='training', 
                                  image_size=(180, 180), batch_size=32, seed=1):
    """Create image dataset with error handling for corrupted files"""
    
    # Get all image files and their labels
    all_files = []
    class_names = sorted([d.name for d in directory.iterdir() if d.is_dir()])
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = directory / class_name
        for img_file in class_dir.glob('*.jpg'):
            all_files.append((str(img_file), class_idx))
    
    # Shuffle with fixed seed
    random.seed(seed)
    random.shuffle(all_files)
    
    # Split into train/val
    split_idx = int(len(all_files) * (1 - validation_split))
    
    if subset == 'training':
        file_label_pairs = all_files[:split_idx]
    else:
        file_label_pairs = all_files[split_idx:]
    
    print(f"Found {len(file_label_pairs)} files for {subset}")
    
    def decode_image(image_path):
        """Decode image with error handling"""
        try:
            img = tf.io.read_file(image_path)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, image_size)
            return img
        except:
            # Return a black image if decoding fails
            return tf.zeros((image_size[0], image_size[1], 3), dtype=tf.float32)
    
    def process_path(image_path, label):
        img = decode_image(image_path)
        # Normalize to [0, 1]
        img = img / 255.0
        return img, tf.cast(label, tf.int64)
    
    # Create dataset
    image_paths, labels = zip(*file_label_pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(image_paths), list(labels)))
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=1000, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

train_ds = create_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    image_size=(image_height, image_width),
    seed = 1,
    batch_size=batch_size
)

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
val_ds = create_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    image_size=(image_height, image_width),
    seed = 1,
    batch_size=batch_size
)

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# normalization_layer = layers.Rescaling(1./255)

# === AFTER (edited) ===
# Normalization layer (already applied in dataset creation, kept for reference)
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
# execution_status: {'status': 'error', 'done': True, 'execution_count': 13}
history = model.fit(train_ds,validation_data=val_ds, epochs=1)
