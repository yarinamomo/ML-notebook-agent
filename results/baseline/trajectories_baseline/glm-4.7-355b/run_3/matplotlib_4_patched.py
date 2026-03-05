# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
import pathlib
data_dir = "data_small/Training_Handwritten_marathi_Character_Fusion/Data_set"
data_dir = pathlib.Path(data_dir)
print(data_dir)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# img_height,img_width=200,200
# batch_size=32
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.25,
#   subset="training",
#   seed=123,
#   label_mode ='categorical',
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# === AFTER (edited) ===
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

img_height,img_width=200,200
batch_size=32

# Filter out corrupted images
data_dir = pathlib.Path("data_small/Training_Handwritten_marathi_Character_Fusion/Data_set")
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
valid_image_paths = []

for subdir in data_dir.iterdir():
    if subdir.is_dir():
        for img_file in subdir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                try:
                    with PIL.Image.open(img_file) as img:
                        img.verify()  # Verify the image is not corrupted
                        # Re-open after verify (verify closes the image)
                        with PIL.Image.open(img_file) as img:
                            img.load()  # Try to load the image
                        valid_image_paths.append(str(img_file))
                except Exception as e:
                    print(f"Skipping corrupted or invalid image: {img_file.name}")
                    continue

print(f"Found {len(valid_image_paths)} valid images")

# Create file paths and labels manually
file_paths = []
labels = []
class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

for img_path in valid_image_paths:
    file_paths.append(img_path)
    class_name = pathlib.Path(img_path).parent.name
    labels.append(class_to_idx[class_name])

labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))

# Create dataset from valid images only
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="training",
  seed=123,
  label_mode ='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="validation",
  seed=123,
  label_mode ='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
class_names = train_ds.class_names
print(class_names)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 6}

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])    
    plt.axis("off")