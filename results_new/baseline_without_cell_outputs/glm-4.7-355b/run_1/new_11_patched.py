# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
import tensorflow as tf
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local")
# strategy = tf.distribute.TPUStrategy(tpu)
# tf.__version__, strategy

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import cv2
import os
import numpy as np

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# class LNClassifier:
#     def __init__(self, base_path, input_size=(224, 224), test_size=0.2, val_size=0.1):
#         self.base_path = base_path
#         self.input_size = input_size
#         self.class_names = []  # Populate with actual class names
#         
#         # Load data first
#         self.X, self.y = self.load_data()
#         
#         # Create train/val/test splits
#         self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(
#             self.X, self.y, 
#             test_size=test_size, 
#             random_state=42, 
#             stratify=self.y
#         )
#         
#         val_ratio = val_size / (1 - test_size)
#         self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
#             self.X_train_val, self.y_train_val,
#             test_size=val_ratio,
#             random_state=42,
#             stratify=self.y_train_val
#         )
#         
#         self.class_weights = class_weight.compute_class_weight(
#             'balanced',
#             classes=np.unique(self.y_train),
#             y=self.y_train
#         )
#         self.class_weight_dict = dict(enumerate(self.class_weights))
# 
#         # Define the data augmentation layers once
#         self.data_augmentation = tf.keras.Sequential([
#             layers.RandomFlip("horizontal"),
#             layers.RandomRotation(0.2),
#             layers.RandomZoom(0.2),
#             layers.RandomContrast(0.2)
#         ])
# 
#         # Move optimizer creation outside the training loop
#         self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.1, nesterov=True)
# 
#     def load_data(self):
#         images = []
#         labels = []
#         
#         for class_id, class_name in enumerate(self.class_names):
#             class_folder = os.path.join(self.base_path, class_name)
#             if os.path.isdir(class_folder):
#                 for filename in os.listdir(class_folder):
#                     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#                         image_path = os.path.join(class_folder, filename)
#                         image = cv2.imread(image_path)
#                         if image is not None:
#                             image = self.preprocess_image(image)
#                             images.append(image)
#                             labels.append(class_id)
#                         else:
#                             print(f"Warning: Could not load image {image_path}")
#             else:
#                 print(f"Warning: {class_folder} does not exist or is not a directory.")
#         
#         print(f"Loaded {len(images)} images across {len(self.class_names)} classes.")
#         return np.array(images), np.array(labels)
# 
#     def preprocess_image(self, image):
#         """Preprocess a single image."""
#         image = cv2.resize(image, self.input_size)
#         image = image / 255.0  # Normalize to [0,1]
#         return image.astype(np.float32)
# 
#     def build_model(self):
#         """Build the model architecture."""
#         base_model = tf.keras.applications.ResNet50(
#             weights='imagenet', 
#             include_top=False, 
#             input_shape=(224, 224, 3)
#         )
#         
#         # Freeze early layers
#         for layer in base_model.layers[:-30]:
#             layer.trainable = False
#         
#         x = base_model.output
#         x = layers.GlobalAveragePooling2D()(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Dense(1024)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Dropout(0.5)(x)
#         output = layers.Dense(len(self.class_names), activation='softmax')(x)
#         
#         return models.Model(inputs=base_model.input, outputs=output)
# 
#     def inspect_dataset(self, dataset):
#         """Inspect the element_spec of a dataset."""
#         print(f"Dataset Element Spec: {dataset.element_spec}")
# 
#     def create_datasets(self, batch_size):
#         """Create tf.data.Dataset objects and inspect them."""
#         train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
#         train_dataset = train_dataset.map(
#             lambda x, y: (self.data_augmentation(x), y)
#         ).batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
#         
#         val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))\
#             .batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
# 
#         # Inspect element_spec
#         self.inspect_dataset(train_dataset)
#         self.inspect_dataset(val_dataset)
# 
#         return train_dataset, val_dataset
# 
#     @tf.function
#     def train_step(self, model, images, labels):
#         """A single training step."""
#         with tf.GradientTape() as tape:
#             predictions = model(images, training=True)
#             loss = SparseCategoricalCrossentropy(from_logits=False)(labels, predictions)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         return loss
# 
#     def train(self, epochs=10, batch_size=32):
#         """Train the model."""
#         # Build and compile model
#         model = self.build_model()
#         model.compile(
#             optimizer=self.optimizer,  # Use the pre-created optimizer here
#             loss=SparseCategoricalCrossentropy(from_logits=False),
#             metrics=['accuracy']
#         )
#         
#         # Create datasets and inspect their structure
#         train_dataset, val_dataset = self.create_datasets(batch_size)
#         
#         # Training loop
#         for epoch in range(epochs):
#             print(f"Epoch {epoch + 1}/{epochs}")
#             for images, labels in train_dataset:
#                 loss = self.train_step(model, images, labels)
#             
#             # Validate after each epoch
#             val_loss, val_accuracy = model.evaluate(val_dataset)
#             print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
# 
#         return model
# 
#     def evaluate_final(self, model):
#         """Evaluate model on the held-out test set."""
#         y_pred = model.predict(self.X_test)
#         y_pred_classes = np.argmax(y_pred, axis=1)
#         
#         print("Test Set Performance:")
#         print("Classification Report:")
#         print(classification_report(self.y_test, y_pred_classes, 
#                                  target_names=self.class_names))
#         
#         print("\nConfusion Matrix:")
#         print(confusion_matrix(self.y_test, y_pred_classes))
#         
#         return y_pred, y_pred_classes

# === AFTER (edited) ===
class LNClassifier:
    def __init__(self, base_path, input_size=(224, 224), test_size=0.2, val_size=0.1):
        self.base_path = base_path
        self.input_size = input_size
        self.class_names = []


        self.X, self.y = self.load_data()


        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=42,
            stratify=self.y
        )

        val_ratio = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_val, self.y_train_val,
            test_size=val_ratio,
            random_state=42,
            stratify=self.y_train_val
        )

        self.class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        self.class_weight_dict = dict(enumerate(self.class_weights))


        self.data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2)
        ])


        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.1, nesterov=True)

    def load_data(self):
        images = []
        labels = []
        
        # Discover class names from directory structure
        if os.path.isdir(self.base_path):
            self.class_names = [d for d in os.listdir(self.base_path) 
                              if os.path.isdir(os.path.join(self.base_path, d))]
            self.class_names.sort()  # Sort for consistent ordering
        else:
            print(f"Warning: {self.base_path} does not exist or is not a directory.")
            return np.array(images), np.array(labels)

        for class_id, class_name in enumerate(self.class_names):
            class_folder = os.path.join(self.base_path, class_name)
            if os.path.isdir(class_folder):
                for filename in os.listdir(class_folder):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(class_folder, filename)
                        image = cv2.imread(image_path)
                        if image is not None:
                            image = self.preprocess_image(image)
                            images.append(image)
                            labels.append(class_id)
                        else:
                            print(f"Warning: Could not load image {image_path}")
            else:
                print(f"Warning: {class_folder} does not exist or is not a directory.")

        print(f"Loaded {len(images)} images across {len(self.class_names)} classes.")
        return np.array(images), np.array(labels)

    def preprocess_image(self, image):
        """Preprocess a single image."""
        image = cv2.resize(image, self.input_size)
        image = image / 255.0
        return image.astype(np.float32)

    def build_model(self):
        """Build the model architecture."""
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )


        for layer in base_model.layers[:-30]:
            layer.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(1024)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(len(self.class_names), activation='softmax')(x)

        return models.Model(inputs=base_model.input, outputs=output)

    def inspect_dataset(self, dataset):
        """Inspect the element_spec of a dataset."""
        print(f"Dataset Element Spec: {dataset.element_spec}")

    def create_datasets(self, batch_size):
        """Create tf.data.Dataset objects and inspect them."""
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_dataset = train_dataset.map(
            lambda x, y: (self.data_augmentation(x), y)
        ).batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))            .batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)


        self.inspect_dataset(train_dataset)
        self.inspect_dataset(val_dataset)

        return train_dataset, val_dataset

    @tf.function
    def train_step(self, model, images, labels):
        """A single training step."""
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = SparseCategoricalCrossentropy(from_logits=False)(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def train(self, epochs=10, batch_size=32):
        """Train the model."""

        model = self.build_model()
        model.compile(
            optimizer=self.optimizer,
            loss=SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )


        train_dataset, val_dataset = self.create_datasets(batch_size)


        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for images, labels in train_dataset:
                loss = self.train_step(model, images, labels)


            val_loss, val_accuracy = model.evaluate(val_dataset)
            print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

        return model

    def evaluate_final(self, model):
        """Evaluate model on the held-out test set."""
        y_pred = model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        print("Test Set Performance:")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred_classes,
                                 target_names=self.class_names))

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred_classes))

        return y_pred, y_pred_classes

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'timeout', 'done': True, 'execution_count': None}
# Initialize classifier
classifier = LNClassifier(
    "data/GLOMERULUS MULTIPLE STAINS",
    test_size=0.2,
    val_size=0.1
)

# Train the model
model, history = classifier.train(epochs=10)

# Evaluate on test set
final_predictions, final_pred_classes = classifier.evaluate_final(model)