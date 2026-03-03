# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# === BEFORE (original) ===
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import skimage.io
# import tensorflow
# import glob
# import tqdm
# import cv2
# from tqdm import tqdm
# 
# from skimage.io import imread, imshow
# from skimage.transform import resize
# 
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization, Dropout, Flatten, Activation
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import torch
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# %matplotlib inline

# === AFTER (edited) ===
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
from PIL import UnidentifiedImageError
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
train_dataset = train_datagen.flow_from_directory(directory = 'data_small/chest-xray-pneumonia/chest_xray/train',
                                                  target_size = (224,224),
                                                  class_mode = 'binary',
                                                  subset = 'training',
                                                  batch_size = 64,
                                                  color_mode='rgb',
                                                  shuffle=True,
                                                  interpolation='bilinear')

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
valid_dataset = valid_datagen.flow_from_directory(directory = 'data_small/chest-xray-pneumonia/chest_xray/train',
                                                  target_size = (224,224),
                                                  class_mode = 'binary',
                                                  subset = 'validation',
                                                  batch_size = 64,
                                                  color_mode='rgb',
                                                  shuffle=False,
                                                  interpolation='bilinear')

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
import warnings
warnings.filterwarnings('ignore')

def safe_fit(model, train_dataset, valid_dataset, epochs, callbacks, verbose=1):
    # Separate earlystopping and checkpoint callbacks
    earlystopping = None
    checkpoint = None
    for callback in callbacks:
        if hasattr(callback, '__class__') and callback.__class__.__name__ == 'EarlyStopping':
            earlystopping = callback
        elif hasattr(callback, '__class__') and callback.__class__.__name__ == 'ModelCheckpoint':
            checkpoint = callback
    
    # Get dataset properties
    steps_per_epoch = len(train_dataset)
    validation_steps = len(valid_dataset)
    
    history = {
        'loss': [],
        'auc': [],
        'val_loss': [],
        'val_auc': []
    }
    
    best_val_auc = -np.inf
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_auc = []
        train_loss = []
        train_batches = 0
        for batch_idx in range(steps_per_epoch):
            try:
                x_batch, y_batch = train_dataset[batch_idx]
                metrics = model.train_on_batch(x_batch, y_batch)
                train_loss.append(metrics[0])
                train_auc.append(metrics[1])
                train_batches += 1
            except (UnidentifiedImageError, OSError, IOError) as e:
                print(f"  Skipping corrupted training batch {batch_idx}")
                continue
        
        if train_batches > 0:
            avg_train_loss = sum(train_loss) / train_batches
            avg_train_auc = sum(train_auc) / train_batches
        else:
            avg_train_loss = 0
            avg_train_auc = 0
        
        # Validation
        val_auc = []
        val_loss = []
        val_batches = 0
        for batch_idx in range(validation_steps):
            try:
                x_batch, y_batch = valid_dataset[batch_idx]
                metrics = model.test_on_batch(x_batch, y_batch)
                val_loss.append(metrics[0])
                val_auc.append(metrics[1])
                val_batches += 1
            except (UnidentifiedImageError, OSError, IOError) as e:
                print(f"  Skipping corrupted validation batch {batch_idx}")
                continue
        
        if val_batches > 0:
            avg_val_loss = sum(val_loss) / val_batches
            avg_val_auc = sum(val_auc) / val_batches
        else:
            avg_val_loss = 0
            avg_val_auc = 0
        
        print(f"  {train_batches}/{steps_per_epoch} - loss: {avg_train_loss:.4f} - auc: {avg_train_auc:.4f} - val_loss: {avg_val_loss:.4f} - val_auc: {avg_val_auc:.4f}")
        
        history['loss'].append(avg_train_loss)
        history['auc'].append(avg_train_auc)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(avg_val_auc)
        
        # Handle checkpoint - manually save best model
        if checkpoint is not None and avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
            try:
                if checkpoint.save_weights_only:
                    model.save_weights(checkpoint.filepath, overwrite=True)
                else:
                    model.save(checkpoint.filepath, overwrite=True)
                if checkpoint.verbose > 0:
                    print(f"Epoch {epoch+1}: val_auc improved from -inf to {avg_val_auc:.5f}, saving model to {checkpoint.filepath}")
            except Exception as e:
                print(f"Warning: Could not save model: {e}")
        
        # Handle early stopping
        if earlystopping is not None:
            if avg_val_auc > best_val_auc:
                best_val_auc = avg_val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= earlystopping.patience:
                    if earlystopping.verbose > 0:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                    break
    
    return type('History', (), {'history': history})()

model_history = safe_fit(model, train_dataset, valid_dataset, epochs=1, callbacks=callback_list, verbose=1)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
class_names = ['PNEUMONIA','NORMAL']

#%%
# --- [CELL 11]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
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

# Use the same indexing approach that yields valid_dataset[index]
for i in range(len(valid_dataset)):
    try:
        x, y = valid_dataset[i]  # This triggers on-the-fly loading using the generator's index handling
        pred = model.predict(x, verbose=0)
        pred_classes = np.argmax(pred, axis=-1)

        y_array = y.numpy() if hasattr(y, 'numpy') else y
        if y_array.ndim > 1:
            y_classes = np.argmax(y_array, axis=-1)
        else:
            y_classes = y_array

        prediction_classes = np.concatenate([prediction_classes, pred_classes])
        true_classes = np.concatenate([true_classes, y_classes])
    except (UnidentifiedImageError, OSError, IOError) as e:
        print(f"Skipping corrupted batch {i}")
        continue

if len(true_classes) > 0 and len(true_classes) == len(prediction_classes):
    print(classification_report(true_classes, prediction_classes, target_names=class_names, digits=4))
else:
    print(f"Warning: not enough samples to evaluate — true: {len(true_classes)}, pred: {len(prediction_classes)}")