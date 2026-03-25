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
for dirname, _, filenames in os.walk('/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
import re
from sklearn.utils import shuffle
# --- 1. LOAD BENGALI DATA ---
path_bengali = "data/BSMDD_v3_textcleaned.csv"
df_bengali = pd.read_csv(path_bengali)
df_bengali = df_bengali[['text', 'label']].dropna()
# Ensure text is string
df_bengali['text'] = df_bengali['text'].astype(str)

# ---------change for reproducing purposes----------
# --- 2. LOAD TWEET DATA (Multilingual) ---
# path_tweets = "/kaggle/input/depression-tweets/master_dataset_7z.csv"
# df_tweets = pd.read_csv(path_tweets, usecols=['text', 'label'])
# df_tweets = df_tweets.dropna()

# --- 3. CLEAN TWEETS ---
def clean_multilingual_tweets(text):
    text = str(text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\S+', '', text)
    # Handle [SEP]
    text = text.replace('[SEP]', ' ')
    # Remove HTML
    text = re.sub(r'<.*?>', '', text)
    # Remove new lines
    text = text.replace('\n', ' ')
    return text.strip()

# print("Cleaning tweets...")
# df_tweets['text'] = df_tweets['text'].apply(clean_multilingual_tweets)

# --- 4. COMBINE & SHUFFLE ---
# ---------change for reproducing purposes----------
df_combined = df_bengali #pd.concat([df_bengali, df_tweets], ignore_index=True)
df_combined = shuffle(df_combined, random_state=42).reset_index(drop=True)
# df_combined = df_bengali
print(f"Total Combined Samples: {len(df_combined)}")
print(df_combined.head())

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
from sklearn.model_selection import train_test_split

# ---------change for reproducing purposes----------
df_combined = df_combined.sample(frac=0.02, random_state=42)

X = df_combined['text'].values
y = df_combined['label'].values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
MAX_LEN = 128
BATCH_SIZE = 32
# Check how many of your texts are longer than MAX_LEN
lengths = [len(text.split()) for text in X_train]
over_limit = sum(1 for x in lengths if x > MAX_LEN)

print(f"Total samples: {len(lengths)}")
print(f"Samples being truncated: {over_limit} ({over_limit/len(lengths)*100:.2f}%)")
print(f"Average length: {np.mean(lengths)}")

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# !pip install -U transformers safetensors
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Conv1D, MaxPooling1D, TimeDistributed, GlobalMaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
MAX_LEN = 128       # Size of each window
MAX_CHUNKS = 8      # Number of windows per text (8 * 128 = 1024 tokens)
BATCH_SIZE = 4      # Small batch size because Sliding Window uses a lot of memory
MODEL_NAME = "sentence-transformers/LaBSE"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# def sliding_window_encode(texts, tokenizer, max_len=MAX_LEN, max_chunks=MAX_CHUNKS):
#     # Total tokens we need to fill 8 windows (8 * 128 = 1024)
#     total_needed = max_len * max_chunks
#     
#     print(f"Tokenizing {len(texts)} samples...")
#     
#     # 1. Tokenize everything at once up to our total limit (1024)
#     # This stops the 512 warning because we are explicitly truncating at 1024
#     encodings = tokenizer(
#         texts.tolist(),
#         add_special_tokens=True,
#         max_length=total_needed,
#         padding='max_length',
#         truncation=True,
#         return_tensors="np"
#     )
#     
#     input_ids = encodings['input_ids']
#     attention_masks = encodings['attention_mask']
#     
#     # 2. Reshape the flat (Samples, 1024) into (Samples, 8, 128)
#     # This automatically chops the 1024 tokens into 8 windows of 128
#     transformed_ids = input_ids.reshape(-1, max_chunks, max_len)
#     transformed_masks = attention_masks.reshape(-1, max_chunks, max_len)
#     
#     return transformed_ids, transformed_masks

# === AFTER (edited) ===
def sliding_window_encode(texts, tokenizer, max_len=MAX_LEN, max_chunks=MAX_CHUNKS):

    num_samples = len(texts)

    print(f"Processing {num_samples} samples for sliding window encoding...")

    encodings = tokenizer(
        texts.tolist(),
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors="np"
    )

    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']

    transformed_ids = input_ids.reshape(-1, 1, max_len)
    transformed_masks = attention_masks.reshape(-1, 1, max_len)

    return transformed_ids, transformed_masks

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# Assuming you have X_train, X_test from your split
print("Starting Sliding Window encoding...")

train_ids, train_masks = sliding_window_encode(X_train, tokenizer)
test_ids, test_masks = sliding_window_encode(X_test, tokenizer)

print(f"New Shape of train_ids: {train_ids.shape}") 
# Result should be: (Samples, 8, 128)
print("Data is now ready for the model.")

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), 
                                 initializer='normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), 
                                 initializer='zeros', trainable=True)        
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
import keras
from keras import ops

# We wrap the LaBSE + CNN logic into a custom layer to avoid Keras 3 "Symbolic" errors
class LaBSEEmbeddingLayer(keras.layers.Layer):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        # Load the model inside the layer
        self.labse_model = TFAutoModel.from_pretrained(model_name, from_pt=True)
        self.labse_model.trainable = False
        
        # Define CNN layers inside this wrapper
        # ---------change for reproducing purposes----------
        self.cnn = Conv1D(64, kernel_size=3, padding='same', activation='relu') # 128
        self.pool = MaxPooling1D(pool_size=2)
        self.global_pool = GlobalMaxPooling1D()

    def call(self, inputs):
        input_ids, input_mask = inputs
        
        # 1. Flatten windows: (Batch, 8, 128) -> (Batch * 8, 128)
        # We use keras.ops.reshape for compatibility
        flat_ids = ops.reshape(input_ids, (-1, MAX_LEN))
        flat_mask = ops.reshape(input_mask, (-1, MAX_LEN))

        # 2. Get LaBSE Embeddings
        embeddings = self.labse_model(flat_ids, attention_mask=flat_mask)[0]

        # 3. Apply CNN
        x = self.cnn(embeddings)
        x = self.pool(x)
        x = self.global_pool(x) # Result is (Batch * 8, 128)

        # 4. Reshape back for LSTM: (Batch * 8, 128) -> (Batch, 8, 128)
        # We use -1 for the batch dimension
        output = ops.reshape(x, (-1, MAX_CHUNKS, 128))
        return output

def build_sliding_window_model():
    # Define Inputs
    input_ids = Input(shape=(MAX_CHUNKS, MAX_LEN), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(MAX_CHUNKS, MAX_LEN), dtype=tf.int32, name="attention_mask")

    # 1. Use the Custom Wrapper Layer
    # This handles the LaBSE call and the CNN in one block
    window_features = LaBSEEmbeddingLayer(MODEL_NAME)([input_ids, input_mask])

    # 2. Bi-LSTM (Processing the sequence of 8 windows)
    # ---------change for reproducing purposes----------
    x = Bidirectional(LSTM(32, return_sequences=True))(window_features) # 64
    
    # 3. Attention (Highlighting the most depressive windows)
    x = Attention()(x)

    # 4. Final Classification Layers
    # ---------change for reproducing purposes----------
    x = Dense(32, activation='relu')(x) # 64
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_ids, input_mask], outputs=outputs)
    return model

# Re-instantiate the model
model = build_sliding_window_model()
model.summary()

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 11}
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('sliding_window_model.keras', save_best_only=True)
]

print("Starting training (Using Sliding Windows)...")
history = model.fit(
    [train_ids, train_masks], y_train,
    validation_data=([test_ids, test_masks], y_test),
    epochs=2, #15,
    batch_size=BATCH_SIZE, # 4
    callbacks=callbacks
)