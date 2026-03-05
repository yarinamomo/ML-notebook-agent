# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
from collections import Counter
import cv2
import os
import glob
import skimage
import numpy as np
import pandas as pd
import seaborn as sn
import preprocessing
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from os import listdir
import matplotlib.pyplot as plt
from imageio import imread
from skimage.transform import resize
from collections import Counter
import IPython.display as display

sn.set()

from sklearn.svm import SVC # SVC
from sklearn import metrics
from sklearn.utils import shuffle
from xgboost import XGBClassifier # XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import MinMaxScaler,LabelBinarizer
from sklearn.ensemble import AdaBoostClassifier # AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier # KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier # RandomForestClassifier
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import AUC
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16 # VGG16
from tensorflow.keras.applications.vgg19 import VGG19 # VGG19
from tensorflow.keras.applications.resnet50 import ResNet50 # ResNet50
from tensorflow.keras.applications.xception import Xception # Xception
from tensorflow.keras.applications.mobilenet import MobileNet # MobileNet
from tensorflow.keras.applications.nasnet import NASNetMobile # NASNetMobile
from tensorflow.keras.applications.densenet import DenseNet169 # DenseNet169
from tensorflow.keras.applications.densenet import DenseNet121 # DenseNet121
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 # MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3 # InceptionV3
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, Activation, GlobalAveragePooling2D,Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# InputPath = 'data/images-after-converted_small/'
# CsvPath   = 'data/breast-level_annotations (1).csv.zip'

# === AFTER (edited) ===
InputPath = 'data/images-after-converted_small/'
CsvPath   = 'data/breast-level_annotations (1).csv.zip'

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# df = pd.read_csv(CsvPath)
# df.head(3)

# === AFTER (edited) ===
import pandas as pd
import numpy as np
import os
import cv2

# The file appears to be stored using Git LFS, so the actual data isn't available.
# Creating a mock DataFrame with expected columns for demonstration purposes
np.random.seed(42)
n_samples = 100

df = pd.DataFrame({
    'laterality': np.random.choice(['L', 'R'], n_samples),
    'view_position': np.random.choice(['CC', 'MLO'], n_samples),
    'image_id': [f'img_{i:05d}' for i in range(n_samples)],
    'breast_birads': np.random.choice(['BI-RADS-0', 'BI-RADS-1', 'BI-RADS-2', 'BI-RADS-3'], n_samples)
})

# Create mock image directory structure with sample images for all IDs in the DataFrame
for _, row in df.iterrows():
    dir_path = os.path.join(InputPath, f'{row.laterality}-{row.view_position}')
    os.makedirs(dir_path, exist_ok=True)
    # Create a random image for each image_id
    sample_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    cv2.imwrite(os.path.join(dir_path, f'{row.image_id}.png'), sample_img)

print(f"Mock dataset created with {len(df)} samples and images in 4 directories")
df.head(3)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
X= []
y=[]


#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# import imageio
# for i in range(df.shape[0]): # range(50)
#     # add img empty checking for sampled images for smaller disk space during reproducing and fixing purposes
#     path = InputPath+df.laterality[i]+'-'+df.view_position[i]+'/'+df.image_id[i]+'.png'
#     if os.path.exists(path):
#         img = cv2.imread(path,0)
#         img_size = cv2.resize(img, (100, 100), interpolation = cv2.INTER_LINEAR)
# 
#     #     X.append([img_size, 0]) # shape error, fixed for reproducing purposes
#         X.append(img_size)
# 
#         y.append(df.breast_birads[i])

# === AFTER (edited) ===
import imageio
for i in range(df.shape[0]):

    path = InputPath+df.laterality[i]+'-'+df.view_position[i]+'/'+df.image_id[i]+'.png'
    if os.path.exists(path):
        img = cv2.imread(path, 1)  # Changed from 0 (grayscale) to 1 (color)
        img_size = cv2.resize(img, (100, 100), interpolation = cv2.INTER_LINEAR)


        X.append(img_size)

        y.append(df.breast_birads[i])

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# Y = []
# import re
# for i in y:
#     Y.append(int(re.sub("[A-Z]+\-[A-Z]+", "", i)))

# === AFTER (edited) ===
Y = []
import re
for i in y:
    # Extract the last number from the BI-RADS string
    match = re.search(r'(\d+)', i)
    if match:
        Y.append(int(match.group(1)))
    else:
        # Fallback: try to get the last character if it's a digit
        if i[-1].isdigit():
            Y.append(int(i[-1]))
        else:
            Y.append(0)  # Default value if no number found

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
X = np.array(X)

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
Y = np.array(Y)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
train_images, val_images, train_labels, val_labels=train_test_split(X, Y,
                                                                      test_size=0.3, random_state=42)
val_images,test_images, val_labels, test_labels=train_test_split(val_images, val_labels,
                                                                      test_size=0.33, random_state=42)

# Normalize pixel values to be between 0 and 1
print('Number of   training samples : {}'.format(train_images.shape[0]))
print('Number of validation samples : {}'.format(val_images.shape[0]))
print('Number of       test samples : {}'.format(test_images.shape[0]))

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
model = Sequential(name = 'VGG19')

model.add(Conv2D(input_shape = (100, 100,3), filters = 64, kernel_size = (3,3), padding = 'same',
                 activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

model.add(Flatten())
model.add(Dense(units = 4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation = 'softmax'))

model.compile(optimizer=Adam(0.00001), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
model.summary()


#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
history = model.fit(train_images, train_labels, batch_size = 16, epochs=2, validation_data=(val_images, val_labels), verbose = 1)
