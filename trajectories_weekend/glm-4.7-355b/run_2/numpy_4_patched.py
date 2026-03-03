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
# cell_state: unchanged
# execution_status: {'execution_count': 2, 'status': 'ok'}
train_datagen = ImageDataGenerator(rescale = 1.0 / 255.0,
                                   zoom_range = 0.4,
                                   validation_split = 0.2)

valid_datagen = ImageDataGenerator(rescale = 1.0 / 255.0,
                                   validation_split = 0.2)

test_datagen  = ImageDataGenerator(rescale = 1.0 / 255.0)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'execution_count': 20, 'status': 'ok'}
# === BEFORE (original) ===
# train_dataset = train_datagen.flow_from_directory(directory = 'data_small/chest-xray-pneumonia/chest_xray/train',
#                                                   target_size = (224,224),
#                                                   class_mode = 'binary',
#                                                   subset = 'training',
#                                                   batch_size = 64)

# === AFTER (edited) ===
import glob
import pandas as pd
import numpy as np
import os
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

class SafeDataGenerator(Sequence):
    """Data generator that handles corrupted images gracefully during runtime"""
    def __init__(self, directory, batch_size, target_size, subset='training', validation_split=0.2):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        
        # Find all image files
        self.files = []
        self.labels = []
        
        for class_name in ['PNEUMONIA', 'NORMAL']:
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                files = glob.glob(os.path.join(class_path, '*.jpeg'))
                self.files.extend(files)
                self.labels.extend([1 if class_name == 'PNEUMONIA' else 0] * len(files))
        
        self.files = np.array(self.files)
        self.labels = np.array(self.labels)
        
        # Create train/validation split based on subset
        if validation_split > 0:
            np.random.seed(42)
            indices = np.random.permutation(len(self.files))
            split_point = int(len(self.files) * (1 - validation_split))
            
            if subset == 'training':
                self.indices = indices[:split_point]
            else:
                self.indices = indices[split_point:]
        else:
            self.indices = np.arange(len(self.files))
        
        print(f"Subset '{subset}': Using {len(self.indices)} samples")
        
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_files = self.files[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        batch_images = []
        valid_labels = []
        
        for img_path, label in zip(batch_files, batch_labels):
            try:
                # Try to load the image
                img = load_img(img_path, target_size=self.target_size)
                img_array = img_to_array(img) / 255.0  # rescale
                
                batch_images.append(img_array)
                valid_labels.append(label)
            except Exception as e:
                # Skip corrupted images
                continue
        
        if len(batch_images) == 0:
            # Return a small dummy batch if all images in batch are corrupted
            # This prevents the training from crashing
            return np.zeros((1, self.target_size[0], self.target_size[1], 3)), np.array([0])
        
        return np.array(batch_images), np.array(valid_labels)

# Create training dataset
train_dataset = SafeDataGenerator(
    directory='data_small/chest-xray-pneumonia/chest_xray/train',
    batch_size=64,
    target_size=(224, 224),
    subset='training',
    validation_split=0.2
)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'execution_count': 21, 'status': 'ok'}
# === BEFORE (original) ===
# valid_dataset = valid_datagen.flow_from_directory(directory = 'data_small/chest-xray-pneumonia/chest_xray/train',
#                                                   target_size = (224,224),
#                                                   class_mode = 'binary',
#                                                   subset = 'validation',
#                                                   batch_size = 64)

# === AFTER (edited) ===
# Create validation dataset
valid_dataset = SafeDataGenerator(
    directory='data_small/chest-xray-pneumonia/chest_xray/train',
    batch_size=64,
    target_size=(224, 224),
    subset='validation',
    validation_split=0.2
)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 5, 'status': 'ok'}
# Defining Model

base_model = VGG16(input_shape=(224,224,3), 
                   include_top=False,
                   weights="imagenet")

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 6, 'status': 'ok'}
for layer in base_model.layers:
    layer.trainable=False

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 7, 'status': 'ok'}
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
# execution_status: {'execution_count': 24, 'status': 'ok'}

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
# cell_state: edited
# execution_status: {'status': 'not run'}
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
  # Get predictions and threshold at 0.5 for binary classification
  preds = model.predict(x)
  pred_classes = (preds > 0.5).astype(int).flatten()
  prediction_classes = np.concatenate([prediction_classes, pred_classes])
  
  # y is already a numpy array, no need for .numpy()
  # For binary classification, labels are already 0 or 1
  true_classes = np.concatenate([true_classes, y.astype(int).flatten()])


print(classification_report(true_classes, prediction_classes, target_names=class_names, digits=4))