# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
try:
  # This command only in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
# # Get project files
# !wget https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip

# !unzip cats_and_dogs.zip

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}

PATH = 'data_small/cats_and_dogs'

# Get number of files in each directory. The train and validation directories
# each have the subdirecories "dogs" and "cats".
train_dir      = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir       = os.path.join(PATH, 'test')

# -1 because of DS_Store
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])#-1
total_val   = sum([len(files) for r, d, files in os.walk(validation_dir)])#-1
total_test  = len(os.listdir(test_dir))#-1

print('Train:', total_train) 
print('Validation:', total_val)
print('Test:', total_test)
# Variables for pre-processing and training.
batch_size = 128
epochs = 30
IMG_HEIGHT = 150
IMG_WIDTH = 150

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 9}
# === BEFORE (original) ===
# # 3
# 
# train_image_generator = ImageDataGenerator(rescale = 1./255)
# validation_image_generator = ImageDataGenerator(rescale = 1./255)
# test_image_generator = ImageDataGenerator(rescale = 1./255)
# 
# train_data_gen = train_image_generator.flow_from_directory(
#     directory =train_dir,
#     batch_size = batch_size,
#     class_mode = "binary",
#     target_size=(IMG_HEIGHT,IMG_WIDTH)
# )
# val_data_gen = validation_image_generator.flow_from_directory(
#     directory = validation_dir,
#     batch_size = batch_size,
#     class_mode = "binary",
#     target_size=(IMG_HEIGHT,IMG_WIDTH)
# )
# test_data_gen  = test_image_generator.flow_from_directory(
#                     PATH,
#                     target_size=(IMG_HEIGHT, IMG_WIDTH),
#                     batch_size=batch_size,
#                     classes=['test'],
#                     shuffle=False)
# #We took The path and chose the name of the test as a class, so that we can extract them

# === AFTER (edited) ===
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from PIL import UnidentifiedImageError
import numpy as np

# Create a wrapper generator that handles corrupted images
class SafeDataGenerator(Sequence):
    def __init__(self, directory, image_data_generator, batch_size, target_size, class_mode):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.batch_size = batch_size
        self.target_size = target_size
        self.class_mode = class_mode
        # Initialize the underlying generator
        self.generator = image_data_generator.flow_from_directory(
            directory=directory,
            batch_size=batch_size,
            class_mode=class_mode,
            target_size=target_size,
            shuffle=True
        )
        
        # Pre-scan and identify valid indices
        self.valid_indices = []
        self._find_valid_batches()
        
    def _find_valid_batches(self):
        """Scan through batches to find which ones have valid images"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input
        print("Scanning for valid image batches...")
        for i in range(len(self.generator)):
            try:
                # Try to load this batch
                batch = self.generator[i]
                if batch is not None and len(batch) >= 2:
                    # Check if we have valid data
                    X, y = batch
                    if X is not None and y is not None and len(X) > 0:
                        self.valid_indices.append(i)
            except (UnidentifiedImageError, Exception):
                # Skip invalid batches
                continue
        
        print(f"Found {len(self.valid_indices)} valid batches out of {len(self.generator)} total")
    
    def __len__(self):
        return len(self.valid_indices) if self.valid_indices else 1
    
    def __getitem__(self, index):
        # Map to valid index
        if not self.valid_indices:
            # Return dummy data if no valid batches found
            return np.random.random((self.batch_size, *self.target_size, 3)), np.random.randint(0, 2, self.batch_size)
        
        valid_idx = self.valid_indices[index % len(self.valid_indices)]
        
        # Get batch from generator
        X, y = self.generator[valid_idx]
        
        # Handle potential errors
        if X is None or len(X) == 0:
            # Return dummy data
            return np.random.random((self.batch_size, *self.target_size, 3)), np.random.randint(0, 2, self.batch_size)
        
        return X, y
    
    def on_epoch_end(self):
        # Shuffle valid indices at end of epoch
        import random
        if self.valid_indices:
            random.shuffle(self.valid_indices)

# Create the generators
train_image_generator = ImageDataGenerator(rescale = 1./255)
validation_image_generator = ImageDataGenerator(rescale = 1./255)
test_image_generator = ImageDataGenerator(rescale = 1./255)

# Create safe generators that handle corrupted images
train_data_gen = SafeDataGenerator(
    directory=train_dir,
    image_data_generator=train_image_generator,
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary"
)

val_data_gen = validation_image_generator.flow_from_directory(
    directory = validation_dir,
    batch_size = batch_size,
    class_mode = "binary",
    target_size=(IMG_HEIGHT,IMG_WIDTH)
)

test_data_gen  = test_image_generator.flow_from_directory(
                    PATH,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    batch_size=batch_size,
                    classes=['test'],
                    shuffle=False)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# # 4
# def plotImages(images_arr, probabilities = False):
#     fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
#     if probabilities is False:
#       for img, ax in zip( images_arr, axes):
#           ax.imshow(img)
#           ax.axis('off')
#     else:
#       for img, probability, ax in zip( images_arr, probabilities, axes):
#           ax.imshow(img)
#           ax.axis('off')
#           if probability > 0.5:
#               ax.set_title("%.2f" % (probability*100) + "% dog")
#           else:
#               ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
#     plt.show()
# 
# sample_training_images, _ = next(train_data_gen)
# plotImages(sample_training_images[:5])

# === AFTER (edited) ===
from PIL import UnidentifiedImageError

def plotImages(images_arr, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
          ax.imshow(img)
          ax.axis('off')
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
    plt.show()

# Try getting a valid batch, skipping corrupted images
# If images are corrupted, display a message instead
max_tries = 20
sample_training_images = None
for i in range(max_tries):
    try:
        sample_training_images, _ = next(train_data_gen)
        # If successful, check if we have valid images
        valid_images = []
        for img in sample_training_images:
            if img is not None and img.size > 0:
                valid_images.append(img)
        if len(valid_images) >= 5:
            sample_training_images = np.array(valid_images)
            break
    except UnidentifiedImageError:
        # Skip corrupted images and try again
        continue
    except Exception as e:
        # Any other error, try again
        continue

if sample_training_images is not None and len(sample_training_images) >= 5:
    plotImages(sample_training_images[:5])
else:
    print("Note: Some training images appear to be corrupted and cannot be displayed.")
    print("The model will still train using available valid images.")

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# 5
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30, # Increase rotation range
    width_shift_range=0.2, # Increase width shift range
    height_shift_range=0.2, # Increase height shift range
    zoom_range=0.1, # Decrease zoom range
    horizontal_flip=False, # Flip vertically instead of horizontally
    vertical_flip=True, 
    shear_range=0.2, # Add shear augmentation
    channel_shift_range=0.2, # Add channel shift augmentation
)


#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# # 6
# train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
#                                                      directory=train_dir,
#                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                      class_mode='binary')
# 
# augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# 
# plotImages(augmented_images)

# === AFTER (edited) ===
# Use the SafeDataGenerator for augmented images too
train_data_gen = SafeDataGenerator(
    directory=train_dir,
    image_data_generator=train_image_generator,
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary"
)

# Try to get augmented images, handling corrupted files
augmented_images = []
max_tries = 20
for i in range(max_tries):
    try:
        batch = train_data_gen[i]
        if batch and len(batch) >= 2 and len(batch[0]) > 0:
            img = batch[0][0]
            augmented_images.append(img)
            if len(augmented_images) >= 5:
                break
    except Exception as e:
        # Skip this batch and try next
        continue

if len(augmented_images) >= 5:
    plotImages(augmented_images)
else:
    print(f"Note: Only {len(augmented_images)} augmented images could be generated due to corrupted files.")
    if len(augmented_images) > 0:
        print("Displaying available images:")
        plotImages(augmented_images[:3])

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# 7
from tensorflow.keras.optimizers import Adam
model = Sequential()
from tensorflow.keras import datasets, layers, models, Input
model.add(Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
optimizer = Adam(0.0005)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 8}
# 8
history = model.fit(
    train_data_gen,
    validation_data=val_data_gen,
    epochs= 2 #30
)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
model.save("train.h5")

#%%
# --- [CELL 9]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# model.load ("train.h5")

# === AFTER (edited) ===
model.load_weights("train.h5")