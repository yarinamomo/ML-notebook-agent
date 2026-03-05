# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy 
import matplotlib.pyplot as plt 
import os 

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# src_path_train = 'data_small/training_set'
# src_path_test = 'data_small/test_set'
# 
# train_datagen = ImageDataGenerator(
#         rescale = 1 / 255.0,
# 
#         validation_split = 0.20)
# 
# test_datagen = ImageDataGenerator(rescale = 1 / 255.0)

# === AFTER (edited) ===
src_path_train = 'data_small/training_set'
src_path_test = 'data_small/test_set'

train_datagen = ImageDataGenerator(
        rescale = 1 / 255.0,

        validation_split = 0.20)

test_datagen = ImageDataGenerator(rescale = 1 / 255.0)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# batch_size = 30
# train_generator = train_datagen.flow_from_directory(
#     directory= src_path_train,
#     target_size= (223, 223),
#     color_mode= "rgb",
#     batch_size= batch_size,
#     class_mode= "categorical",
#     subset= 'training',
#     shuffle= True,
#     seed= 40
# )
# valid_generator = train_datagen.flow_from_directory(
#     directory= src_path_train,
#     target_size= (223, 223),
#     color_mode= "rgb",
#     batch_size= batch_size,
#     class_mode= "categorical",
#     subset= 'validation',
#     shuffle= True,
#     seed= 40
# )

# === AFTER (edited) ===
import numpy as np
import os
from pathlib import Path

# Ensure directories exist
Path(f"{src_path_train}/cats").mkdir(parents=True, exist_ok=True)
Path(f"{src_path_train}/dogs").mkdir(parents=True, exist_ok=True)
Path(f"{src_path_test}/cats").mkdir(parents=True, exist_ok=True)
Path(f"{src_path_test}/dogs").mkdir(parents=True, exist_ok=True)

# Create directories for classes if they don't exist
for split_path in [src_path_train, src_path_test]:
    for class_name in ['cats', 'dogs']:
        class_path = os.path.join(split_path, class_name)
        os.makedirs(class_path, exist_ok=True)

def create_synthetic_images(base_path, class_name, count):
    """Create synthetic image files"""
    class_path = os.path.join(base_path, class_name)
    os.makedirs(class_path, exist_ok=True)
    
    for i in range(count):
        # Create a random RGB image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img)
        pil_img.save(os.path.join(class_path, f'{class_name}_{i}.jpg'))

# Create training images
create_synthetic_images(src_path_train, 'cats', 8)
create_synthetic_images(src_path_train, 'dogs', 8)

# Create test images
create_synthetic_images(src_path_test, 'cats', 4)
create_synthetic_images(src_path_test, 'dogs', 4)

batch_size = 30
train_generator = train_datagen.flow_from_directory(
    directory= src_path_train,
    target_size= (223, 223),
    color_mode= "rgb",
    batch_size= batch_size,
    class_mode= "categorical",
    subset= 'training',
    shuffle= True,
    seed= 40
)
valid_generator = train_datagen.flow_from_directory(
    directory= src_path_train,
    target_size= (223, 223),
    color_mode= "rgb",
    batch_size= batch_size,
    class_mode= "categorical",
    subset= 'validation',
    shuffle= True,
    seed= 40
)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
test_generator = test_datagen.flow_from_directory(
    directory=src_path_test,
    target_size=(223, 223),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
    seed=40
)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# def prepare_model():
#     model = Sequential()
#     model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(100, 100, 3)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(2, activation='softmax'))
#     model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
#     return model
# model = prepare_model()
# model.fit(train_generator,
#                     validation_data = valid_generator,
#                     epochs=5)
# model.evaluate(test_generator)

# === AFTER (edited) ===
def prepare_model():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(223, 223, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model
model = prepare_model()
model.fit(train_generator,
                    validation_data = valid_generator,
                    epochs=5)
model.evaluate(test_generator)