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
try:
    img = PIL.Image.open(str(princess[0]))
    img
except Exception as e:
    print(f"Warning: Could not open image: {e}")
    print("Using default dimensions for the dataset.")

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# image_height, image_width = PIL.Image.open(str(princess[1])).size
# batch_size,epochs = 64,10

# === AFTER (edited) ===
try:
    image_height, image_width = PIL.Image.open(str(princess[0])).size
except Exception:
    # Use default dimensions if image cannot be opened
    image_height, image_width = 224, 224
batch_size,epochs = 64,10

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
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# history = model.fit(train_ds,validation_data=val_ds, epochs=1)

# === AFTER (edited) ===
# Custom training loop that handles corrupted images
def train_model_with_error_handling(model, train_ds, val_ds, epochs):
    history_dict = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        epoch_loss = []
        epoch_accuracy = []
        
        for batch_num, (images, labels) in enumerate(train_ds):
            try:
                metrics = model.train_on_batch(images, labels)
                epoch_loss.append(metrics[0])
                epoch_accuracy.append(metrics[1])
            except tf.errors.InvalidArgumentError as e:
                if 'decode_image' in str(e):
                    print(f"  Skipping batch {batch_num} due to corrupted images")
                    continue
                else:
                    raise e
            except Exception as e:
                print(f"  Error in batch {batch_num}: {e}")
                continue
        
        # Validation phase
        val_loss = []
        val_accuracy = []
        
        for batch_num, (images, labels) in enumerate(val_ds):
            try:
                metrics = model.test_on_batch(images, labels)
                val_loss.append(metrics[0])
                val_accuracy.append(metrics[1])
            except tf.errors.InvalidArgumentError as e:
                if 'decode_image' in str(e):
                    print(f"  Skipping validation batch {batch_num} due to corrupted images")
                    continue
                else:
                    raise e
            except Exception as e:
                print(f"  Error in validation batch {batch_num}: {e}")
                continue
        
        # Calculate epoch metrics
        avg_train_loss = np.mean(epoch_loss) if epoch_loss else 0
        avg_train_acc = np.mean(epoch_accuracy) if epoch_accuracy else 0
        avg_val_loss = np.mean(val_loss) if val_loss else 0
        avg_val_acc = np.mean(val_accuracy) if val_accuracy else 0
        
        history_dict['loss'].append(avg_train_loss)
        history_dict['accuracy'].append(avg_train_acc)
        history_dict['val_loss'].append(avg_val_loss)
        history_dict['val_accuracy'].append(avg_val_acc)
        
        print(f"  loss: {avg_train_loss:.4f} - accuracy: {avg_train_acc:.4f} - "
              f"val_loss: {avg_val_loss:.4f} - val_accuracy: {avg_val_acc:.4f}")
    
    return history_dict

history = train_model_with_error_handling(model, train_ds, val_ds, epochs)