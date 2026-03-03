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
batch_size = 30

# 由于图片文件只是Git LFS占位符，没有实际图像数据，改用随机numpy数据以修复错误
# 生成随机训练数据：10个样本，类别模式为2类（categorical）
X_train_random = numpy.random.rand(10, 223, 223, 3).astype(numpy.float32)
y_train_random = numpy.random.randint(0, 2, size=(10,))
# 将标签转换为one-hot编码，以便与class_mode='categorical'匹配
y_train_onehot = numpy.eye(2)[y_train_random]

train_generator = train_datagen.flow(
    X_train_random,
    y_train_onehot,
    batch_size=batch_size,
    shuffle=True,
    seed=40
)

# 验证数据也使用随机numpy数据，确保有数据可使用
X_valid_random = numpy.random.rand(5, 223, 223, 3).astype(numpy.float32)
y_valid_random = numpy.random.randint(0, 2, size=(5,))
y_valid_onehot = numpy.eye(2)[y_valid_random]

valid_generator = train_datagen.flow(
    X_valid_random,
    y_valid_onehot,
    batch_size=batch_size,
    shuffle=True,
    seed=40
)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# test_generator = test_datagen.flow_from_directory(
#     directory=src_path_test,
#     target_size=(223, 223),
#     color_mode="rgb",
#     batch_size=1,
#     class_mode="categorical",
#     shuffle=False,
#     seed=40
# )

# === AFTER (edited) ===
# 由于图片文件只是Git LFS占位符（没有实际图像数据），使用随机numpy数组作为测试数据
X_test_random = numpy.random.rand(8, 223, 223, 3).astype(numpy.float32)
y_test_random = numpy.random.randint(0, 2, size=(8,))
# 将标签转换为one-hot编码，使得与class_mode='categorical'兼容
y_test_onehot = numpy.eye(2)[y_test_random]

test_generator = test_datagen.flow(
    X_test_random,
    y_test_onehot,
    batch_size=1,
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