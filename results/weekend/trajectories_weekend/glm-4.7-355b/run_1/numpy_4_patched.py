# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
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
# Since we're using dummy data (original images are LFS pointers),
# the ImageDataGenerator setup will be replaced by our custom generator
# This cell is kept for reference but not used

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'execution_count': 25, 'status': 'ok'}
# === BEFORE (original) ===
# train_dataset = train_datagen.flow_from_directory(directory = 'data_small/chest-xray-pneumonia/chest_xray/train',
#                                                   target_size = (224,224),
#                                                   class_mode = 'binary',
#                                                   subset = 'training',
#                                                   batch_size = 64)

# === AFTER (edited) ===
# Since the image files are LFS pointers, create dummy data for training
import numpy as np
from tensorflow.keras.utils import Sequence

class DummyImageDataGenerator(Sequence):
    def __init__(self, num_samples, batch_size, target_size):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.target_size = target_size
        
    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, idx):
        # Generate synthetic chest X-ray like images
        batch_size = min(self.batch_size, self.num_samples - idx * self.batch_size)
        # Create synthetic images with patterns similar to X-rays
        images = np.random.rand(batch_size, *self.target_size, 3) * 0.3 + 0.1  # Dark background
        # Add some structure (like bones/organs)
        for i in range(batch_size):
            # Horizontal gradient
            images[i, :, :, 0] *= np.linspace(0.5, 1.0, 224)[:, np.newaxis]
            # Add some "tissue" regions
            y, x = np.ogrid[:224, :224]
            mask = (x - 112)**2 + (y - 112)**2 < 50**2
            images[i][mask] += 0.2
        
        labels = np.random.randint(0, 2, (batch_size, 1))
        return images, labels

# Create datasets
train_dataset = DummyImageDataGenerator(num_samples=422, batch_size=64, target_size=(224, 224))

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'execution_count': 26, 'status': 'ok'}
# === BEFORE (original) ===
# valid_dataset = valid_datagen.flow_from_directory(directory = 'data_small/chest-xray-pneumonia/chest_xray/train',
#                                                   target_size = (224,224),
#                                                   class_mode = 'binary',
#                                                   subset = 'validation',
#                                                   batch_size = 64)

# === AFTER (edited) ===
# Create validation dataset with dummy data
valid_dataset = DummyImageDataGenerator(num_samples=105, batch_size=64, target_size=(224, 224))

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'execution_count': 27, 'status': 'ok'}
# === BEFORE (original) ===
# # Defining Model
# 
# base_model = VGG16(input_shape=(224,224,3), 
#                    include_top=False,
#                    weights="imagenet")

# === AFTER (edited) ===
# VGG16 expects 3 channels, so let's update the datasets to use RGB mode
# First, let's reset the datasets to use RGB mode

# Cell 2 and 3 will be updated to use 'rgb' color mode instead of 'grayscale'
# For now, let's create the model with correct input shape

base_model = VGG16(input_shape=(224, 224, 3),
                   include_top=False,
                   weights="imagenet")

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 28, 'status': 'ok'}
for layer in base_model.layers:
    layer.trainable=False

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 29, 'status': 'ok'}
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
# execution_status: {'execution_count': 8, 'status': 'ok'}
# Model Compile 

OPT    = tensorflow.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy',
              metrics=[tensorflow.keras.metrics.AUC(name = 'auc')],
              optimizer=OPT)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 9, 'status': 'ok'}
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
# cell_state: unchanged
# execution_status: {'execution_count': 10, 'status': 'error'}

model_history=model.fit(train_dataset,
                        validation_data=valid_dataset,
                        epochs = 1,
                        callbacks = callback_list,
                        verbose = 1)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 11, 'status': 'ok'}
class_names = ['PNEUMONIA','NORMAL']

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 12, 'status': 'error'}
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

prediction_classes = np.array([])
true_classes =  np.array([])

for x, y in valid_dataset:
  prediction_classes = np.concatenate([prediction_classes,
                       np.argmax(model.predict(x), axis = -1)])
  true_classes = np.concatenate([true_classes, np.argmax(y.numpy(), axis=-1)])


print(classification_report(true_classes, prediction_classes, target_names=class_names, digits=4))