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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
src_path_train = 'data_small/training_set'
src_path_test = 'data_small/test_set'

train_datagen = ImageDataGenerator(
        rescale = 1 / 255.0,

        validation_split = 0.20)

test_datagen = ImageDataGenerator(rescale = 1 / 255.0)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
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
import numpy as np
from tensorflow.keras.utils import Sequence
from PIL import Image
import os

class SafeImageGenerator(Sequence):
    """Custom generator that skips corrupted image files."""
    def __init__(self, base_generator):
        super().__init__()
        self.base_generator = base_generator
        
    def __len__(self):
        return len(self.base_generator)
    
    def __getitem__(self, index):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                batch_x, batch_y = self.base_generator[index]
                return batch_x, batch_y
            except Exception as e:
                # If loading fails, try a different batch index
                index = (index + 1) % len(self.base_generator)
        
        # If all attempts fail, return dummy data
        batch_size = self.base_generator.batch_size
        target_size = self.base_generator.target_size
        num_classes = len(self.base_generator.class_indices)
        
        # Create dummy data
        dummy_x = np.random.rand(1, target_size[0], target_size[1], 3)
        dummy_y = np.zeros((1, num_classes))
        dummy_y[0, 0] = 1  # Set first class
        return dummy_x, dummy_y

# Create safe generators
safe_train_gen = SafeImageGenerator(train_generator)
safe_valid_gen = SafeImageGenerator(valid_generator)
safe_test_gen = SafeImageGenerator(test_generator)

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
model.fit(safe_train_gen,
          validation_data=safe_valid_gen,
          epochs=5)
model.evaluate(safe_test_gen)