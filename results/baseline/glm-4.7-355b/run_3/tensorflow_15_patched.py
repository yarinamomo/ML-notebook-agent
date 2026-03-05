# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
from pathlib import Path
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import subprocess
from IPython.display import FileLink, display

from sklearn.metrics import confusion_matrix, classification_report

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# # Count number of training images for both classes to calculate a
# # data-driven training batch size.
# num_samples = (len(os.listdir('data_small/Chic')) +
#                len(os.listdir('data_small/Duck')))
# 
# # We use 200 batches.
# img_height, img_width = 224,224
# batch_size = num_samples // 200

# === AFTER (edited) ===
import numpy as np
import pandas as pd
from pathlib import Path
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import subprocess
from IPython.display import FileLink, display

from sklearn.metrics import confusion_matrix, classification_report

from PIL import Image

# Function to remove corrupted or unsupported image files
def clean_image_directory(directory_path):
    """Remove image files that cannot be opened with PIL"""
    if not os.path.exists(directory_path):
        return
    
    removed_files = []
    for class_folder in os.listdir(directory_path):
        class_path = os.path.join(directory_path, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Verify the image
                    # Re-open to verify again (verify() closes the file)
                    with Image.open(img_path) as img:
                        img.load()
                except Exception as e:
                    try:
                        os.remove(img_path)
                        removed_files.append(img_path)
                    except:
                        pass
    
    if removed_files:
        print(f"Removed {len(removed_files)} corrupted/unsupported image files from {directory_path}")
    
    return len(removed_files) if removed_files else 0

# Clean the datasets before using them
print("Cleaning 'data_small' directory...")
clean_image_directory('data_small')
print("Cleaning 'data_small_test' directory...")
clean_image_directory('data_small_test')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 3}
train_ds = tf.keras.utils.image_dataset_from_directory(
  'data_small',
  validation_split=0.2,
  subset="training",
  label_mode='binary',
  seed=123, #number to randomize outcome
  image_size=(img_height, img_width),
  batch_size=batch_size)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
val_ds = tf.keras.utils.image_dataset_from_directory(
 'data_small',
  validation_split=0.2,
  subset="validation",
  label_mode='binary',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
test_ds = tf.keras.utils.image_dataset_from_directory(
 'data_small_test',
  image_size=(img_height, img_width),
  label_mode='binary',
  batch_size=batch_size)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
base_model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)

# 1024 neurons are half the 2048 output dimensionality of the previous
# layer and the last layer from base ResNet-50.
x = keras.layers.Dense(units=1024, activation='relu')(x)
x = keras.layers.Dense(units=512, activation='relu')(x)
x = keras.layers.Dense(units=256, activation='relu')(x)
x = keras.layers.Dense(units=128, activation='relu')(x)
x = keras.layers.Dense(units=1, activation='sigmoid')(x)

model = keras.models.Model(inputs=base_model.input,
                                    outputs=x)

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
for layer in model.layers[:175]:
    layer.trainable = False

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 10}
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2, #100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=3
        )
    ]
)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
results = model.evaluate(test_ds, verbose=0)
print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
predictions = (model.predict(test_ds) >= 0.5)

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
predictions = np.array([])
labels =  np.array([])
for x, y in test_ds:
  predictions = np.concatenate([predictions, model.predict_classes(x)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()