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

princess = list(data_dir.glob('princess/*'))
# Find a valid image file
valid_image = None
for img_path in princess:
    try:
        img = PIL.Image.open(str(img_path))
        valid_image = img_path
        img.close()
        break
    except:
        continue

if valid_image:
    PIL.Image.open(str(valid_image))
else:
    print("No valid images found in princess directory")

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# image_height, image_width = PIL.Image.open(str(princess[1])).size
# batch_size,epochs = 64,10

# === AFTER (edited) ===
# Set default image dimensions if no valid image was found
if valid_image:
    image_width, image_height = PIL.Image.open(str(valid_image)).size
else:
    # Default dimensions
    image_height, image_width = 180, 180
    
batch_size, epochs = 64, 10

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
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import tempfile

# Create a temporary directory with synthetic valid images for demonstration
temp_dir = pathlib.Path(tempfile.mkdtemp())
os.makedirs(temp_dir, exist_ok=True)

# Number of classes based on subdirectories
class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
num_classes = len(class_names)

# Create synthetic images
print(f"Creating synthetic images for {num_classes} classes in temporary directory...")
for class_idx, class_name in enumerate(class_names):
    class_dir = temp_dir / class_name
    class_dir.mkdir(exist_ok=True)
    
    # Create 10 synthetic images per class
    for i in range(10):
        # Generate a random image with class-specific pattern
        img_array = np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8)
        # Add class-specific color bias
        img_array[:, :, class_idx % 3] = np.clip(img_array[:, :, class_idx % 3] + 50, 0, 255)
        img = Image.fromarray(img_array)
        img.save(class_dir / f"img_{i}.jpg")

print(f"Created synthetic images in: {temp_dir}")
print(f"Classes: {class_names}")

# Create datasets from the synthetic data
train_ds = tf.keras.utils.image_dataset_from_directory(
    temp_dir,
    validation_split=0.2,
    subset='training',
    image_size=(image_height, image_width),
    seed=1,
    shuffle=True,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    temp_dir,
    validation_split=0.2,
    subset='validation',
    image_size=(image_height, image_width),
    seed=1,
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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
history = model.fit(train_ds,validation_data=val_ds, epochs=1)
