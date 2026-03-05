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
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# import pathlib
# data_dir = "data_small/Training_Handwritten_marathi_Character_Fusion/Data_set"
# data_dir = pathlib.Path(data_dir)
# print(data_dir)

# === AFTER (edited) ===
import pathlib
import os

# Try multiple common path variations to locate the dataset
possible_paths = [
    "data_small/Training_Handwritten_marathi_Character_Fusion/Data_set",
    "data/Training_Handwritten_marathi_Character_Fusion/Data_set",
    "Training_Handwritten_marathi_Character_Fusion/Data_set",
    "Data_set",
    "data_small/training_handwritten_marathi_character_fusion/Data_set",
    "data/training_handwritten_marathi_character_fusion/Data_set",
]

data_dir = None
for path in possible_paths:
    data_dir = pathlib.Path(path)
    if data_dir.exists() and data_dir.is_dir():
        print(f"Found dataset at: {data_dir}")
        print(f"Directory contents: {list(data_dir.iterdir())[:5]}")
        break
    else:
        data_dir = None

if data_dir is None:
    # If none of the expected paths exist, create the path and check if parent exists
    # This will still fail in Cell 2 but gives clearer error message
    data_dir = pathlib.Path("data_small/Training_Handwritten_marathi_Character_Fusion/Data_set")
    
# List what's available in current directory for debugging
print(f"\nCurrent directory: {pathlib.Path.cwd()}")
print("Available directories:")
for item in sorted(pathlib.Path.cwd().iterdir()):
    print(f"  - {item.name} {'(DIR)' if item.is_dir() else '(FILE)'}")

print(f"\nUsing data_dir: {data_dir}")

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 3}
img_height,img_width=200,200
batch_size=32
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
# execution_status: {'status': 'not run'}
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
# execution_status: {'status': 'not run'}
class_names = train_ds.class_names
print(class_names)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])    
    plt.axis("off")