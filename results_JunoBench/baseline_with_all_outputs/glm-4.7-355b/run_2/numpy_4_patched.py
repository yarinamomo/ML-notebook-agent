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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
train_datagen = ImageDataGenerator(rescale = 1.0 / 255.0,
                                   zoom_range = 0.4,
                                   validation_split = 0.2)

valid_datagen = ImageDataGenerator(rescale = 1.0 / 255.0,
                                   validation_split = 0.2)

test_datagen  = ImageDataGenerator(rescale = 1.0 / 255.0)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
train_dataset = train_datagen.flow_from_directory(directory = 'data_small/chest-xray-pneumonia/chest_xray/train',
                                                  target_size = (224,224),
                                                  class_mode = 'binary',
                                                  subset = 'training',
                                                  batch_size = 64)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
valid_dataset = valid_datagen.flow_from_directory(directory = 'data_small/chest-xray-pneumonia/chest_xray/train',
                                                  target_size = (224,224),
                                                  class_mode = 'binary',
                                                  subset = 'validation',
                                                  batch_size = 64)

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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
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
  # For binary classification, convert probabilities to binary predictions
  pred_probs = model.predict(x, verbose=0)
  pred_binary = (pred_probs > 0.5).astype(int).flatten()
  prediction_classes = np.concatenate([prediction_classes, pred_binary])
  # y is already a numpy array of 0s and 1s for binary classification
  true_classes = np.concatenate([true_classes, y])

print(classification_report(true_classes, prediction_classes, target_names=class_names, digits=4))