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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# image_count = len(list(data_dir.glob('*/*.jpg')))
# image_count

# === AFTER (edited) ===
import PIL

def is_valid_image(path):
    try:
        with PIL.Image.open(path) as img:
            img.verify()
        # Re-open to check if it can be loaded
        with PIL.Image.open(path) as img:
            img.load()
        return True
    except:
        return False

# Get all image paths
all_image_paths = list(data_dir.glob('*/*.jpg'))

# Filter out corrupted images
valid_image_paths = [p for p in all_image_paths if is_valid_image(p)]
image_count = len(valid_image_paths)
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
# Create a custom filtered dataset to avoid corrupt files
def create_filtered_dataset(data_dir, subset):
    # Collect image paths and labels
    image_paths = []
    labels = []
    
    class_names = sorted([str(d.name) for d in data_dir.iterdir() if d.is_dir()])
    
    valid_count = 0
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        for img_path in class_dir.glob('*.jpg'):
            # Validate image
            try:
                with PIL.Image.open(img_path) as img:
                    img.verify()
                # Re-open to ensure it loads
                with PIL.Image.open(img_path) as img:
                    img.load()
                valid_count += 1
            except:
                continue
            
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    print(f"Found {len(image_paths)} valid images out of {valid_count + (40 - len(image_paths))} total")
    
    # Shuffle indices
    indices = np.random.RandomState(seed=1).permutation(len(image_paths))
    
    # Split based on subset
    split_idx = int(0.8 * len(indices))
    if subset == 'training':
        indices = indices[:split_idx]
    else:
        indices = indices[split_idx:]
    
    # Create dataset from filtered images
    image_paths = [image_paths[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    # Define function to load and preprocess image
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [image_height, image_width])
        return img, label
    
    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000, seed=1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

train_ds = create_filtered_dataset(data_dir, 'training')

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
val_ds = create_filtered_dataset(data_dir, 'validation')

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
# Count number of classes from the data directory
num_of_classes = len([d for d in data_dir.iterdir() if d.is_dir()])
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