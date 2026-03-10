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
import os

class_names = sorted(os.listdir(data_dir))
valid_files = []

# Check each image file for validity
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            try:
                with PIL.Image.open(file_path) as img:
                    img.verify()  # Verify the image is valid and not corrupted
                    valid_files.append(class_name)
            except Exception as e:
                print(f"Skipping invalid image: {file_path} - {e}")

# Create filtered dataset
file_list = []
label_list = []
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            try:
                with PIL.Image.open(file_path) as img:
                    img.load()  # Load to verify
                    file_list.append(file_path)
                    label_list.append(class_names.index(class_name))
            except Exception as e:
                print(f"Skipping invalid image: {file_path} - {e}")

# Create dataset programmatically
def load_images(file_paths, labels):
    images = []
    final_labels = []
    for path, label in zip(file_paths, labels):
        try:
            img = tf.keras.preprocessing.image.load_img(
                path, target_size=(image_height, image_width)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            final_labels.append(label)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    return np.array(images), np.array(final_labels)

# Load all images
train_images, train_labels = load_images(file_list, label_list)

# Create tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(buffer_size=len(train_images)).batch(batch_size)

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
# Split the loaded data into train and validation sets
num_train = int(0.8 * len(train_images))
num_val = len(train_images) - num_train

train_images_final = train_images[:num_train]
train_labels_final = train_labels[:num_train]

val_images_final = train_images[num_train:]
val_labels_final = train_labels[num_train:]

# Create training dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_images_final, train_labels_final))
train_ds = train_ds.shuffle(buffer_size=len(train_images_final)).batch(batch_size)

# Create validation dataset
val_ds = tf.data.Dataset.from_tensor_slices((val_images_final, val_labels_final))
val_ds = val_ds.batch(batch_size)

print(f"Training samples: {len(train_images_final)}, Validation samples: {len(val_images_final)}")

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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
# === BEFORE (original) ===
# num_of_classes = len(train_ds.class_names)
# num_of_classes

# === AFTER (edited) ===
num_of_classes = len(class_names)
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