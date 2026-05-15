# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import skimage.io
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
train_datagen = ImageDataGenerator(#rotation_range = 180,
                                         width_shift_range = 0.1,
                                         height_shift_range = 0.1,
                                         horizontal_flip = True,
                                         rescale = 1./255,
                                         #zoom_range = 0.2,
                                         validation_split = 0.2
                                        )
valid_datagen = ImageDataGenerator(rescale = 1./255,
                                         validation_split = 0.2)
test_datagen = ImageDataGenerator(rescale = 1./255,
                                         validation_split = 0.2)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
train_dataset=train_datagen.flow_from_directory(directory='data_small/train',
                                               target_size=(48,48),
                                               class_mode='categorical',
                                               subset='training',
                                               batch_size=64)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
valid_dataset=valid_datagen.flow_from_directory(directory='data_small/test',
                                               target_size=(48,48),
                                               class_mode='categorical',
                                               batch_size=64)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
test_dataset=test_datagen.flow_from_directory(directory='data_small/test',
                                               target_size=(48,48),
                                               class_mode='categorical',
                                               batch_size=64)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
train_datagen = ImageDataGenerator(
    # Other parameters...
    rotation_range=20,  # Example: Add rotation for augmentation
    zoom_range=0.2  # Example: Add zooming for augmentation
)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# the fxies here are for reproducing purposes
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU  
from tensorflow.keras.regularizers import l2 

inputs=Input((64,64,3))

h=Conv2D(64,(1,1),padding='same',activation='relu')(inputs)
h=BatchNormalization()(h)
h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=Conv2D(64,(3,3))(h)
h=BatchNormalization()(h)
#     h=MaxPooling2D((2,2),strides=(2,2))(h)
h=Activation('relu')(h)

b=Conv2D(128,(1,1),strides=(2,2))(h)
b=BatchNormalization()(b)

h=SeparableConv2D(128,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=Activation('relu')(h)
h=SeparableConv2D(128,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h=MaxPooling2D((2,2),strides=(2,2))(h)

h=concatenate([h,b],name='first')

b=Conv2D(128,(2,2),strides=(2,2))(h)
b=BatchNormalization()(b)

h=SeparableConv2D(128,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=SeparableConv2D(128,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h=MaxPooling2D((2,2),strides=(2,2))(h)

h=concatenate([h,b],name='second')

b=Conv2D(256,(1,1),padding='same')(h)
b=BatchNormalization()(b)
b=MaxPooling2D((2,2),strides=(2,2))(b)

h=SeparableConv2D(256,(3,3),padding='same')(h)
h=BatchNormalization()(h)
# h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=Activation('relu')(h)
h=SeparableConv2D(256,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h=MaxPooling2D((2,2),strides=(2,2))(h)

h=concatenate([h,b],name='third')
b=h

h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=SeparableConv2D(512,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=SeparableConv2D(512,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=SeparableConv2D(512,(3,3),padding='same')(h)
h=BatchNormalization()(h)

h=concatenate([h,b],name='fourth')

b=Conv2D(512,(1,1),padding='same')(h)
b=BatchNormalization()(b)
b=MaxPooling2D((2,2),strides=(2,2))(b)

h=SeparableConv2D(512,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=SeparableConv2D(512,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h=MaxPooling2D((2,2),strides=(2,2))(h)

h=concatenate([h,b],name='fifth')

b=Conv2D(1024,(1,1),padding='same')(h)
b=BatchNormalization()(b)
b=MaxPooling2D((2,2),strides=(2,2))(b)

h=SeparableConv2D(1024,(3,3),padding='same')(h)
h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=SeparableConv2D(1024,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h=MaxPooling2D((2,2),strides=(2,2))(h)

h=concatenate([h,b],name='sixth')
b=h

h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)

h=SeparableConv2D(512,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=SeparableConv2D(256,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=SeparableConv2D(128,(3,3),padding='same')(h)
h=BatchNormalization()(h)

h=concatenate([h,b],name='seventh')
b=h

b=Conv2D(256,(1,1),strides=(1,1))(h)
b=BatchNormalization()(b)

h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=SeparableConv2D(1024,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)
h=SeparableConv2D(512,(3,3),padding='same')(h)
h=BatchNormalization()(h)

h=concatenate([h,b],name='eighth')

h=SeparableConv2D(256,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h = Activation('relu')(h)   # fix # h=tf.nn.relu(h)



x = GlobalAveragePooling2D()(h)

# Fully Connected Layer
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.4)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.4)(x)

# Additional Fully Connected Layer
x = Dense(256)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.3)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.2)(x)


# Output Layer
outputs = Dense(7, activation='softmax')(x)



# Create the model
model = Model(inputs=inputs, outputs=outputs)
model.summary()

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
def f1_score(y_true,y_pred):
    true_positives=K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    possible_positives=K.sum(K.round(K.clip(y_true,0,1)))
    predicted_positives=K.sum(K.round(K.clip(y_pred,0,1)))
    precision=true_positives/(predicted_positives+K.epsilon())
    recall=true_positives/(possible_positives+K.epsilon())
    f1_val=2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
METRICS=[
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
      f1_score,
]

#%%
# --- [CELL 9]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 10}
# === BEFORE (original) ===
# from sklearn.utils.class_weight import compute_class_weight
# class_weights = compute_class_weight('balanced', np.unique(train_dataset.labels), train_dataset.labels)
# class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
# 
# # Compile the model with class weights
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=METRICS, class_weight=class_weight_dict)

# === AFTER (edited) ===
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', np.unique(train_dataset.labels), train_dataset.labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=METRICS)