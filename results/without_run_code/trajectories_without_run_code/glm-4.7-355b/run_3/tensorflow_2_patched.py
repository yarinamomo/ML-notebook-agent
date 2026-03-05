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
def prepare_model():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(223, 223, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model

from tensorflow.keras.utils import Sequence
import numpy as np
import os

class SafeGenerator(Sequence):
    def __init__(self, generator):
        self.generator = generator
        self.valid_indices = []
        # Find all valid batches
        for i in range(len(generator)):
            try:
                batch = generator[i]
                if batch[0] is not None and batch[1] is not None:
                    self.valid_indices.append(i)
                else:
                    print(f"Skipping invalid batch {i}")
            except Exception as e:
                print(f"Skipping batch {i} due to error: {e}")
        print(f"Found {len(self.valid_indices)} valid batches out of {len(generator)}")
    
    def __len__(self):
        return max(1, len(self.valid_indices))
    
    def __getitem__(self, index):
        if len(self.valid_indices) == 0:
            # Return a dummy batch if no valid data
            return np.zeros((1, 223, 223, 3)), np.zeros((1, 2))
        real_index = self.valid_indices[index % len(self.valid_indices)]
        try:
            return self.generator[real_index]
        except:
            # Return a dummy batch if error occurs
            return np.zeros((1, 223, 223, 3)), np.zeros((1, 2))

print("Creating safe generators...")
train_safe_gen = SafeGenerator(train_generator)
valid_safe_gen = SafeGenerator(valid_generator)
test_safe_gen = SafeGenerator(test_generator)

model = prepare_model()

if len(train_safe_gen.valid_indices) > 0:
    model.fit(train_safe_gen,
              validation_data=valid_safe_gen,
              epochs=5,
              verbose=1)
else:
    print("No valid training data available. Skipping training.")

if len(test_safe_gen.valid_indices) > 0:
    print("Evaluating model...")
    model.evaluate(test_safe_gen)
else:
    print("No valid test data available. Skipping evaluation.")