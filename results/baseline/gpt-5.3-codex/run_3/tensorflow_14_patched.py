# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# Define VGG_FACE_MODEL architecture
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))	
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# Load VGG Face model weights
model.load_weights('data/vgg_face_weights.h5')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# # Set the main data directory where subdirectories represent classes/labels
# main_data_directory = 'data/train-data-imgs'
# 
# # Define the input size for the VGG16 model
# input_size = (224, 224)
# 
# # Create a data generator for training data
# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     zoom_range=0.2
# )
# 
# train_generator = train_datagen.flow_from_directory(
#     main_data_directory,
#     target_size=input_size,
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=True
# )
# 
# # Load the VGG16 model without the top classification layer
# base_model = VGG16(weights='imagenet', include_top=False,classes=7)
# 
# # Make the layers in the base model non-trainable
# for layer in base_model.layers:
#     layer.trainable = False
# 
# # Compile the model with an appropriate optimizer, loss function, and metrics
# base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 
# # Load the previously saved model weights
# base_model.load_weights('data/vgg_face_weights.h5')
# 
# # Continue training the model
# base_model.fit(
#     train_generator,
#     steps_per_epoch=len(train_generator),
#     epochs=10,  # You can adjust the number of epochs
# )
# 
# # Save the model after additional training
# base_model.save('data/updated_vgg_face_weights.h5')

# === AFTER (edited) ===
main_data_directory = 'data/train-data-imgs'

input_size = (224, 224)

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    main_data_directory,
    target_size=input_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Use ImageNet VGG16 as feature extractor for 7-class training
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Removed incompatible VGG-Face weight loading (caused layer count mismatch)

base_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
)

base_model.save('data/updated_vgg_face_weights.h5')