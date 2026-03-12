# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Average, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
np.random.seed(123)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
data_dir = 'data_small/CovidDataset_70_30'

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
batch_size = 32
input_shape = (224, 224, 3)
num_classes = 2

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'Train'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'Test'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        os.path.join(data_dir, 'Validation'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical')

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

modelV19 = Sequential()
modelV19.add(vgg19)
modelV19.add(Flatten())
modelV19.add(Dense(500, activation='relu'))
modelV19.add(Dropout(0.5))
modelV19.add(Dense(300, activation='relu'))
modelV19.add(Dropout(0.5))
modelV19.add(Dense(num_classes, activation='softmax'))

for layer in vgg19.layers:
    layer.trainable = False
# modelV19.summary()

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
mobilenetv2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

modelM2 = Sequential()
modelM2.add(mobilenetv2)
modelM2.add(Flatten())
modelM2.add(Dense(500, activation='relu'))
modelM2.add(Dropout(0.5))
modelM2.add(Dense(300, activation='relu'))
modelM2.add(Dropout(0.5))
modelM2.add(Dense(num_classes, activation='softmax'))

for layer in mobilenetv2.layers:
    layer.trainable = False
    
# modelM2.summary()

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
opt = Adam(learning_rate=0.0001, beta_1=0.9)
opt2 = Adam(learning_rate=0.0001, beta_1=0.9) # fix for reproducing and fixing purposes, need a new optimizer instance
modelV19.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])

modelM2.compile(
    loss='binary_crossentropy',
    optimizer=opt2,
    metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10)
filepath_weights_V19 = "data_small/best_weights_V19-{epoch:02d}-{val_accuracy:.4f}.keras"
filepath_weights_M2 = "data_small/best_weights_M2-{epoch:02d}-{val_accuracy:.4f}.keras"
checkpoint_V19 = ModelCheckpoint(filepath_weights_V19, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
checkpoint_M2 = ModelCheckpoint(filepath_weights_M2, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
# Dont run again model is saved @ /kaggle/working/save_weights/best_weights_V19-47-0.9566.hdf5

history_V19 = modelV19.fit(train_generator, epochs=2, validation_data=validation_generator, callbacks=[early_stop, checkpoint_V19])

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
# Dont run again model is saved @ /kaggle/working/save_weights/best_weights_M2-49-0.9681.hdf5
history_M2 = modelM2.fit(train_generator, epochs=2, validation_data=validation_generator, callbacks=[early_stop, checkpoint_M2])

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average

# fix for reproducing and fixing purposes - because the best model names maybe different every running and they are hardcoded now
# model_1 = load_model('/kaggle/working/save_weights/best_weights_V19-47-0.9566.hdf5')
# model_2 = load_model('/kaggle/working/save_weights/best_weights_M2-49-0.9681.hdf5')

# # Set layers in model_1 and model_2 to be not trainable
# for layer in model_1.layers:
#     layer.trainable = False

# for layer in model_2.layers:
#     layer.trainable = False
    
    
# model_1 = Model(
#     inputs = model_1.inputs,
#     outputs = model_1.outputs,
#     name = "VGG16"
# )

# model_2 = Model(
#     inputs = model_2.inputs,
#     outputs = model_2.outputs,
#     name = "MobileNetV2"
# )

# models = [model_1,model_2]
models = [modelV19, modelM2]
# fix ends

model_input = Input(shape=(224, 224, 3))
model_outputs = [model(model_input) for model in models]

ensemble_output = Average()(model_outputs)


ensemble_model = Model(
    inputs = model_input,
    outputs = ensemble_output,
    name = "Ensemble"
)

# ensemble_model.summary()

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
opt3 = Adam(learning_rate=0.0001, beta_1=0.9) # fix for reproducing and fixing purposes, need a new optimizer instance
# only compile when there are trainable parameters
ensemble_model.compile(
    loss='binary_crossentropy',
    optimizer=opt3,
    metrics=['accuracy'])

filepath_weights_ensemble = "data_small/best_weights_ensemble-{epoch:02d}-{val_accuracy:.4f}.keras"
checkpoint_ensemble = ModelCheckpoint(filepath_weights_ensemble, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
# only fit when there are trainable parameters
history_ensemble = ensemble_model.fit(train_generator, epochs=2, validation_data=validation_generator, callbacks=[early_stop, checkpoint_ensemble])

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
# ensemble_model.load_weights('/kaggle/working/save_weights/best_weights_ensemble-01-0.9752.tf')
# test_loss, test_acc = ensemble_model.evaluate(test_generator)
# print('Test accuracy:', test_acc)
ensemble_model.evaluate(test_generator)

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
y_pred = ensemble_model.predict(test_generator)
print("One-hot encoded predicted labels:")
print(y_pred)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred_classes)

#%%
# --- [CELL 15]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 16}
# === BEFORE (original) ===
# #ensemble_model.load_weights('/kaggle/working/save_weights/best_weights_ensemble-31-0.9690.tf')
# #y_pred = ensemble_model.predict(test_generator)
# #y_pred_classes = np.argmax(y_pred, axis=1)
# #y_true_classes = test_generator.classes
# 
# #try this 
# # Make predictions on test data
# y_pred = ensemble_model.predict(test_generator)
# y_pred_classes = np.argmax(y_pred, axis=1)
# 
# # Convert one-hot encoded labels to integer labels
# y_true_onehot = test_generator.classes
# y_true_classes = np.argmax(y_true_onehot, axis=1)

# === AFTER (edited) ===
y_pred = ensemble_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true_classes = test_generator.classes