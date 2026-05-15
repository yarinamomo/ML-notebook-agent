# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os 
from os import listdir
from tqdm import tqdm
import shutil


import tensorflow as tf
from tensorflow import keras
from keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
#from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten,BatchNormalization, GlobalAveragePooling2D 
#from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Reshape, Dropout, Concatenate, AvgPool2D, MaxPool2D, DepthwiseConv2D, ChannelShuffle
from keras.utils import to_categorical
from keras.preprocessing import image

from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense, AveragePooling2D, MaxPooling2D, BatchNormalization, Conv2D, GlobalAveragePooling2D, Lambda,Permute,ZeroPadding2D,DepthwiseConv2D,Reshape,Concatenate


from keras.callbacks import LearningRateScheduler
from keras import backend as K


%matplotlib inline
#************************************************

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
size=75

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
benign_images = getListOfFiles('data/augmented/benign')
malignent_images = getListOfFiles('data/augmented/malignant')

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
data = pd.DataFrame(index=np.arange(0, len(benign_images)+len(malignent_images)), columns=["image", "target"])
k=0

for c in [0,1]:
        if c==1:
            for m in range(len(benign_images)):
                data.iloc[k]["image"] = benign_images[m]
                data.iloc[k]["target"] = 0
                k += 1
        else:
            for m in range(len(malignent_images)):
                data.iloc[k]["image"] = malignent_images[m]
                data.iloc[k]["target"] = 1
                k += 1

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
ben_upsampled = resample(data[data['target']==0],n_samples=data[data['target']==1].shape[0], random_state=42)

up_sampled = pd.concat([data[data['target']==1], ben_upsampled])

up_sampled['target'].value_counts()

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# train_image = []
# y = []
# 
# for i in tqdm(range(up_sampled.shape[0])):
#     img = tf.keras.utils.load_img(up_sampled['image'].iloc[i], target_size=(size,size,1), color_mode="grayscale")
#     img = tf.keras.utils.img_to_array(img)
#     img = img/255
#     train_image.append(img)
# 
#         
# X = np.array(train_image)
# y = up_sampled.iloc[:,-1].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=42, test_size=0.2 , shuffle=True)
# 
# Y_train = to_categorical(y_train, 2)
# Y_test = to_categorical(y_test, 2)
# Y_val = to_categorical(y_val, 2)
# 
# print(X_train.shape)
# print(X_test.shape)
# print(X_val.shape)

# === AFTER (edited) ===
train_image = []
y = []

for i in tqdm(range(up_sampled.shape[0])):
    img = tf.keras.utils.load_img(up_sampled['image'].iloc[i], target_size=(size,size,3), color_mode="rgb")
    img = tf.keras.utils.img_to_array(img)
    img = img/255
    train_image.append(img)


X = np.array(train_image)
y = up_sampled.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=42, test_size=0.2 , shuffle=True)

Y_train = to_categorical(y_train, 2)
Y_test = to_categorical(y_test, 2)
Y_val = to_categorical(y_val, 2)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
def channel_shuffle(x, groups):
    """
    Channel shuffle operation as described in the ShuffleNet paper.

    Args:
        x: Input tensor.
        groups: Number of groups to divide the channels into.

    Returns:
        Channel shuffled tensor.
    """
    def shuffle_op(x):
        batch_size, height, width, channels = x.shape.as_list()
        channels_per_group = channels // groups
        x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [-1, height, width, channels])
        return x

    return Lambda(shuffle_op)(x)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
def shuffle_unit(x, in_channels, out_channels, bottleneck_channels):
    # define residual
    res = x

    # 1x1 group convolution
    x = Conv2D(filters=bottleneck_channels, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,groups=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # channel shuffle
    x = Lambda(channel_shuffle, arguments={'groups': 2})(x)

    # 1x3 depthwise convolution
    x = DepthwiseConv2D(kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    # 3x1 depthwise convolution
    x = DepthwiseConv2D(kernel_size=(3, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    # 1x1 group convolution
    x = Conv2D(filters=out_channels, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,groups=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # concatenate with the residual connection
    if in_channels != out_channels:
        res = Conv2D(filters=out_channels, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(res)
        res = BatchNormalization()(res)
        
    x = Concatenate()([x, res])
    x = Activation('relu')(x)
    
    return x

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
def GridSizeReductionBlock(inputs, filters):
    # Max pooling layer
    path1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)

    # Convolution layer 1
    path2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)

    # Convolution layer 2
    path3 = Conv2D(filters=filters//2, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
    path3 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(path3)

    # Concatenate the paths
    output = Concatenate()([path1, path2, path3])
    return output

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
inputs = Input(shape=(size, size, 3))

# initial layers
x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
# shuffle units
#res = Conv2D(256, (1, 1), padding='same', use_bias=False, name='shortcut_conv')(x)
#res = BatchNormalization(name='shortcut_bn')(res)

# shuffle unit
x = shuffle_unit(x, in_channels=64, out_channels=256, bottleneck_channels=64)
x = shuffle_unit(x, in_channels=64, out_channels=256, bottleneck_channels=64)
x = shuffle_unit(x, in_channels=64, out_channels=256, bottleneck_channels=64)

x = GridSizeReductionBlock(x, filters=256)

x = shuffle_unit(x, in_channels=128, out_channels=512, bottleneck_channels=128)
x = shuffle_unit(x, in_channels=128, out_channels=512, bottleneck_channels=128)
x = shuffle_unit(x, in_channels=128, out_channels=512, bottleneck_channels=128)
x = shuffle_unit(x, in_channels=128, out_channels=512, bottleneck_channels=128)
x = GridSizeReductionBlock(x, filters=512)

x = shuffle_unit(x, in_channels=256, out_channels=1024, bottleneck_channels=256)
x = shuffle_unit(x, in_channels=256, out_channels=1024, bottleneck_channels=256)
x = shuffle_unit(x, in_channels=256, out_channels=1024, bottleneck_channels=256)
x = shuffle_unit(x, in_channels=256, out_channels=1024, bottleneck_channels=256)
x = shuffle_unit(x, in_channels=256, out_channels=1024, bottleneck_channels=256)
x = shuffle_unit(x, in_channels=256, out_channels=1024, bottleneck_channels=256)
x = GridSizeReductionBlock(x, filters=1024)

x = shuffle_unit(x, in_channels=512, out_channels=2048, bottleneck_channels=512)
x = shuffle_unit(x, in_channels=512, out_channels=2048, bottleneck_channels=512)
x = shuffle_unit(x, in_channels=512, out_channels=2048, bottleneck_channels=512)
x = GridSizeReductionBlock(x, filters=2048)

# add more shuffle units here...




# final layers
x = GlobalAveragePooling2D()(x)
x = Dense(units=2, activation='softmax')(x)

# create model
model = Model(inputs=inputs, outputs=x)

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
data_gen = ImageDataGenerator(
        zoom_range=1.2,  # set range for random zoom
        rotation_range = 90,
        width_shift_range=0.5,
        height_shift_range=0.5,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )
data_gen.fit(X_train)

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
history = model.fit(data_gen.flow(X_train,Y_train,
                                       batch_size=64, 
                                       seed=27,
                                       shuffle=True),
                    
                    epochs=2,
                    validation_data=(X_test,Y_test),
                    verbose=1)