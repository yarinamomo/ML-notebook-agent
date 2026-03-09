# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import os 
import time
import matplotlib.pyplot as plt

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# Generation resolution - Must be square 
# Training data is also scaled to this.
# Note GENERATE_RES 4 or higher  
# will blow Google CoLab's memory and have not
# been tested extensivly.
GENERATE_RES = 2 # Generation resolution factor 
# (1=32, 2=64, 3=96, 4=128, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
IMAGE_CHANNELS = 3

# Preview image 
PREVIEW_ROWS = 1
PREVIEW_COLS = 1
PREVIEW_MARGIN = 16

# Size vector to generate images from
SEED_SIZE = 100

# Configuration
DATA_PATH = 'data'
EPOCHS = 5000
BATCH_SIZE = 32
BUFFER_SIZE = 60000

print(f"Will generate {GENERATE_SQUARE}px square images.")

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# training_binary_path = os.path.join(DATA_PATH,
#         f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')
# 
# print(f"Looking for file: {training_binary_path}")
# 
# if not os.path.isfile(training_binary_path):
#   start = time.time()
#   print("Loading training images...")
# 
#   training_data = []
#   faces_path = 'data/apple_disease_classification/Train/Blotch_Apple'
#   for filename in tqdm(os.listdir(faces_path)):
#       path = os.path.join(faces_path,filename)
#       image = Image.open(path).resize((GENERATE_SQUARE,
#             GENERATE_SQUARE),Image.LANCZOS)
#       training_data.append(np.asarray(image))
#   training_data = np.reshape(training_data,(-1,GENERATE_SQUARE,
#             GENERATE_SQUARE,3))
#   training_data = training_data.astype(np.float32)
#   training_data = training_data / 127.5 - 1.
# 
# 
#   print("Saving training image binary...")
# #   np.save(training_binary_path,training_data)
#   elapsed = time.time()-start
#   
# else:
#   print("Loading previous training pickle...")
#   training_data = np.load(training_binary_path)

# === AFTER (edited) ===
training_binary_path = os.path.join(DATA_PATH,
        f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')

print(f"Looking for file: {training_binary_path}")

if not os.path.isfile(training_binary_path):
  start = time.time()
  print("Loading training images...")

  training_data = []
  faces_path = 'data/apple_disease_classification/Train/Blotch_Apple'
  for filename in tqdm(os.listdir(faces_path)):
      path = os.path.join(faces_path,filename)
      # Convert to RGB to ensure all images have 3 channels
      image = Image.open(path).convert('RGB').resize((GENERATE_SQUARE,
            GENERATE_SQUARE),Image.LANCZOS)
      training_data.append(np.asarray(image))
  training_data = np.reshape(training_data,(-1,GENERATE_SQUARE,
            GENERATE_SQUARE,3))
  training_data = training_data.astype(np.float32)
  training_data = training_data / 127.5 - 1.


  print("Saving training image binary...")

  elapsed = time.time()-start

else:
  print("Loading previous training pickle...")
  training_data = np.load(training_binary_path)