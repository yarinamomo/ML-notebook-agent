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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
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
train_datagen = ImageDataGenerator(rescale = 1.0 / 255.0,
                                   zoom_range = 0.4,
                                   validation_split = 0.2)

valid_datagen = ImageDataGenerator(rescale = 1.0 / 255.0,
                                   validation_split = 0.2)

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
import os
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from PIL import Image
import io

# Custom Sequence that safely loads images with error handling
class SafeImageDataSequence(Sequence):
    def __init__(self, directory, batch_size=64, target_size=(224, 224), 
                 shuffle=True, subset='training', validation_split=0.2):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.subset = subset
        self.validation_split = validation_split
        
        # Get class directories
        self.class_names = sorted([d for d in os.listdir(directory) 
                                 if os.path.isdir(os.path.join(directory, d))])
        self.class_indices = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Build file list
        self.filepaths = []
        self.labels = []
        for class_name in self.class_names:
            class_dir = os.path.join(directory, class_name)
            for fname in os.listdir(class_dir):
                filepath = os.path.join(class_dir, fname)
                self.filepaths.append(filepath)
                self.labels.append(self.class_indices[class_name])
        
        # Split data
        num_samples = len(self.filepaths)
        if self.subset == 'training':
            indices = np.arange(num_samples)[:int(num_samples * (1 - validation_split))]
        else:
            indices = np.arange(num_samples)[int(num_samples * (1 - validation_split)):]
        
        self.filepaths = [self.filepaths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        if self.shuffle:
            self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.filepaths) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.filepaths))
        
        batch_filepaths = self.filepaths[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        batch_images = []
        valid_labels = []
        
        for filepath, label in zip(batch_filepaths, batch_labels):
            try:
                # Try to load and process image
                with open(filepath, 'rb') as f:
                    img_bytes = f.read()
                img = Image.open(io.BytesIO(img_bytes))
                img = img.convert('RGB')
                img = img.resize(self.target_size)
                img_array = np.array(img, dtype=np.float32) / 255.0
                batch_images.append(img_array)
                valid_labels.append(label)
            except Exception as e:
                print(f"Warning: Skipping corrupted image {filepath}: {str(e)}")
                continue
        
        if len(batch_images) == 0:
            # Return a dummy batch to avoid empty errors
            batch_images = np.zeros((1, self.target_size[0], self.target_size[1], 3))
            valid_labels = np.zeros(1)
        else:
            batch_images = np.array(batch_images)
            valid_labels = np.array(valid_labels, dtype=np.float32)
        
        return batch_images, valid_labels
    
    def on_epoch_end(self):
        if self.shuffle:
            indices = np.random.permutation(len(self.filepaths))
            self.filepaths = [self.filepaths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

# Create train dataset
train_dataset = SafeImageDataSequence(
    directory='data_small/chest-xray-pneumonia/chest_xray/train',
    batch_size=64,
    target_size=(224, 224),
    shuffle=True,
    subset='training',
    validation_split=0.2
)

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
# Create validation dataset
valid_dataset = SafeImageDataSequence(
    directory='data_small/chest-xray-pneumonia/chest_xray/train',
    batch_size=64,
    target_size=(224, 224),
    shuffle=False,
    subset='validation',
    validation_split=0.2
)

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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
# === BEFORE (original) ===
# 
# model_history=model.fit(train_dataset,
#                         validation_data=valid_dataset,
#                         epochs = 1,
#                         callbacks = callback_list,
#                         verbose = 1)

# === AFTER (edited) ===
model_history=model.fit(train_dataset,
                        validation_data=valid_dataset,
                        epochs = 1,
                        callbacks = callback_list,
                        verbose = 1)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
class_names = ['PNEUMONIA','NORMAL']

#%%
# --- [CELL 11]: ---
# cell_state: edited
# execution_status: {'status': 'timeout', 'done': True, 'execution_count': 12}
# === BEFORE (original) ===
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# 
# prediction_classes = np.array([])
# true_classes =  np.array([])
# 
# for x, y in valid_dataset:
#   prediction_classes = np.concatenate([prediction_classes,
#                        np.argmax(model.predict(x), axis = -1)])
#   true_classes = np.concatenate([true_classes, np.argmax(y.numpy(), axis=-1)])
# 
# 
# print(classification_report(true_classes, prediction_classes, target_names=class_names, digits=4))

# === AFTER (edited) ===
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

prediction_classes = np.array([])
true_classes =  np.array([])

for x, y in valid_dataset:
  predictions = model.predict(x)
  # For binary classification with sigmoid output, use threshold
  pred_labels = np.round(predictions).flatten()
  prediction_classes = np.concatenate([prediction_classes, pred_labels])
  # Labels are already numpy arrays, no .numpy() needed, and binary so no argmax
  true_classes = np.concatenate([true_classes, y])


print(classification_report(true_classes, prediction_classes, target_names=class_names, digits=4))