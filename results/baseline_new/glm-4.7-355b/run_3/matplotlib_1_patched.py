# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import tensorflow as tf
from tensorflow import keras

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images.shape, train_labels.shape, test_images.shape, test_labels.shape

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# import matplotlib.pyplot as plt
# 
# class_names = "airplane automobile bird cat deer dog frog horse ship truck".split()
# 
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.imshow(train_images[i].astype("uint8"))
#     plt.title(class_names[train_labels[i]])
#     plt.axis("off")

# === AFTER (edited) ===
import matplotlib.pyplot as plt

class_names = "airplane automobile bird cat deer dog frog horse ship truck".split()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(train_images[i].astype("uint8"))
    plt.title(class_names[train_labels[i, 0]])
    plt.axis("off")