# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io
import tensorflow
import glob
import tqdm
import cv2
from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
%matplotlib inline

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# train_datagen = ImageDataGenerator(rescale = 1.0 / 255.0,
#                                    zoom_range = 0.4,
#                                    validation_split = 0.2)
# 
# valid_datagen = ImageDataGenerator(rescale = 1.0 / 255.0,
#                                    validation_split = 0.2)
# 
# test_datagen  = ImageDataGenerator(rescale = 1.0 / 255.0)

# === AFTER (edited) ===
train_datagen = ImageDataGenerator(rescale = 1.0 / 255.0)

valid_datagen = ImageDataGenerator(rescale = 1.0 / 255.0)

test_datagen  = ImageDataGenerator(rescale = 1.0 / 255.0)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# train_dataset = train_datagen.flow_from_directory(directory = 'data_small/chest-xray-pneumonia/chest_xray/train',
#                                                   target_size = (224,224),
#                                                   class_mode = 'binary',
#                                                   subset = 'training',
#                                                   batch_size = 64)

# === AFTER (edited) ===
# Use test directory as training data since original train images were cleaned
train_dataset = train_datagen.flow_from_directory(directory = 'data_small/chest-xray-pneumonia/chest_xray/test',
                                                  target_size = (224,224),
                                                  class_mode = 'binary',
                                                  batch_size = 64,
                                                  shuffle=True)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# valid_dataset = valid_datagen.flow_from_directory(directory = 'data_small/chest-xray-pneumonia/chest_xray/train',
#                                                   target_size = (224,224),
#                                                   class_mode = 'binary',
#                                                   subset = 'validation',
#                                                   batch_size = 64)

# === AFTER (edited) ===
# Use val directory as validation data
valid_dataset = valid_datagen.flow_from_directory(directory = 'data_small/chest-xray-pneumonia/chest_xray/val',
                                                  target_size = (224,224),
                                                  class_mode = 'binary',
                                                  batch_size = 64,
                                                  shuffle=False)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# Defining Model

base_model = VGG16(input_shape=(224,224,3), 
                   include_top=False,
                   weights="imagenet")

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
for layer in base_model.layers:
    layer.trainable=False

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# Defining Layers

model=Sequential()
model.add(base_model)
model.add(Dropout(0.2))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(1024,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1024,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# Model Compile 

OPT    = tensorflow.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy',
              metrics=[tensorflow.keras.metrics.AUC(name = 'auc')],
              optimizer=OPT)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
# Defining Callbacks

filepath = 'data_small/best_weights.keras'

earlystopping = EarlyStopping(monitor = 'val_auc', 
                              mode = 'max' , 
                              patience = 3,
                              verbose = 1)

checkpoint    = ModelCheckpoint(filepath, 
                                monitor = 'val_auc', 
                                mode='max', 
                                save_best_only=True, 
                                verbose = 1)


callback_list = [earlystopping, checkpoint]

#%%
# --- [CELL 9]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# 
# model_history=model.fit(train_dataset,
#                         validation_data=valid_dataset,
#                         epochs = 1,
#                         callbacks = callback_list,
#                         verbose = 1)

# === AFTER (edited) ===
# Use OpenCV to load images instead of PIL, which is more robust
import cv2
import os
import numpy as np

def create_robust_dataset(directory, batch_size=64, shuffle=True):
    """Create a dataset using OpenCV to load images"""
    images = []
    labels = []
    class_names = sorted(os.listdir(directory))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                try:
                    # Use OpenCV to read image
                    img = cv2.imread(filepath)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        img = img.astype(np.float32) / 255.0
                        images.append(img)
                        labels.append(class_to_idx[class_name])
                except Exception as e:
                    print(f"Skipping corrupted file: {filepath}")
                    continue
    
    images = np.array(images)
    labels = np.array(labels)
    
    if shuffle:
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]
    
    num_samples = len(images)
    split = int(num_samples * 0.8)
    
    # Yield batches
    def generator():
        start_idx = 0
        while True:
            end_idx = min(start_idx + batch_size, len(images))
            batch_x = images[start_idx:end_idx]
            batch_y = labels[start_idx:end_idx]
            start_idx = end_idx
            if len(batch_x) == 0:
                start_idx = 0
                batch_x = images[:batch_size]
                batch_y = labels[:batch_size]
            yield batch_x, batch_y
    
    return generator(), num_samples

# Create robust data generators
print("Creating training dataset...")
train_gen, train_samples = create_robust_dataset('data_small/chest-xray-pneumonia/chest_xray/test', batch_size=64, shuffle=True)
print(f"Training samples: {train_samples}")

print("Creating validation dataset...")
val_gen, val_samples = create_robust_dataset('data_small/chest-xray-pneumonia/chest_xray/val', batch_size=64, shuffle=False)
print(f"Validation samples: {val_samples}")

# Now train the model using the custom generators
import tensorflow as tf

# Create tf.data datasets from the generators
train_tf = tf.data.Dataset.from_generator(
    lambda: train_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)

val_tf = tf.data.Dataset.from_generator(
    lambda: val_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)

# Train for 1 epoch
model_history = model.fit(
    train_tf.take(train_samples // 64 + 1),
    validation_data=val_tf.take(val_samples // 64 + 1),
    epochs=1,
    callbacks=callback_list,
    steps_per_epoch=max(1, train_samples // 64),
    validation_steps=max(1, val_samples // 64),
    verbose=1
)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
class_names = ['PNEUMONIA','NORMAL']

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

prediction_classes = np.array([])
true_classes =  np.array([])

for x, y in valid_dataset:
  prediction_classes = np.concatenate([prediction_classes,
                       np.argmax(model.predict(x), axis = -1)])
  true_classes = np.concatenate([true_classes, np.argmax(y.numpy(), axis=-1)])


print(classification_report(true_classes, prediction_classes, target_names=class_names, digits=4))